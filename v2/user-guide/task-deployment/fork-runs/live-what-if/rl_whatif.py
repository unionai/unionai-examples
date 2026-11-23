# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.13.4",
#    "flyteplugins-union>=0.24.0",
# ]
# main = "train"
# params = "gen_latency_s=0"
# ///

"""What-if analysis on a running RL job: branch training from step k with a different reward,
while the original keeps going.

A GRPO-style loop trains a policy to answer math questions. Each step generates a group of
rollouts, grades them with a reward function, updates the policy with group-normalized
advantages, and evaluates accuracy on a held-out set. Every step is its own `train_step` action
that takes the training state (policy, learning rate, reward weights) and returns the next one,
so the run's action tree is a list of checkpoints.

The reward has a bug that's easy to ship: it gives a bonus for longer answers ("reward showing
your work"). The policy finds that rambling earns more reward than reasoning, so reward climbs
while accuracy falls and answers get three times longer. That's reward hacking, and it usually
shows up a few steps into a long, expensive run.

With a live fork you don't stop the run or restart from scratch. While it's still training, edit
the `state` input of the step where you want to branch (lower `length_bonus`) and fork. Every
step before it is reused, the fork trains on from that checkpoint with the new reward, and the
source keeps training on the old one, so the two reward and accuracy curves come out side by
side in the runs' reports.

The policy is a softmax over three answering behaviors with fixed accuracy and length, so the
example runs on a CPU in seconds per step, with no model or GPU. In a real job, `generate` is a
vLLM batch on GPUs and `update` is the optimizer step, which is exactly why reusing them matters.

Run:
  python rl_whatif.py                        # launch training, print how to fork it
  python rl_whatif.py branch <run-name> 4    # branch at step 4 with length_bonus=0.1, from Python
"""

import asyncio
import math
import random
import statistics
import sys
from dataclasses import dataclass

import flyte
import flyte.remote as remote

if __name__ == "__main__":
    # Only the machine that forks needs the plugin; the task image doesn't install it.
    from flyteplugins.union import fork

image = flyte.Image.from_debian_base(python_version=(3, 13), install_flyte=False).with_pip_packages(
    "flyte>=2.13.4"
)

env = flyte.TaskEnvironment(
    name="rl_whatif",
    image=image,
    resources=flyte.Resources(cpu="250m", memory="500Mi"),
)

# The three ways the policy can answer: (accuracy, mean answer length in tokens).
BEHAVIORS = {"reason": (0.80, 320), "guess": (0.25, 40), "ramble": (0.30, 1400)}
NAMES = list(BEHAVIORS)


# {{docs-fragment state}}
@dataclass
class TrainState:
    """Everything a training step needs. Editing it on one step's action branches training there."""

    step: int
    logits: list[float]  # the policy: one logit per behavior
    lr: float
    length_bonus: float  # reward per 1,000 tokens of answer: the bug
    group_size: int


@dataclass
class Rollout:
    behavior: str
    correct: bool
    length: int


@dataclass
class StepResult:
    state: TrainState  # the next step's input
    reward: float
    accuracy: float
    mean_length: float
# {{/docs-fragment state}}


def _probs(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    return [x / sum(e) for x in e]


@env.task
async def generate(logits: list[float], step: int, group_size: int, latency_s: int) -> list[Rollout]:
    """Sample a group of rollouts from the policy. In a real job: a vLLM batch on GPUs."""
    await asyncio.sleep(latency_s)
    rng = random.Random(f"{step}-{[round(x, 6) for x in logits]}")
    rollouts = []
    for _ in range(group_size):
        name = rng.choices(NAMES, weights=_probs(logits))[0]
        accuracy, length = BEHAVIORS[name]
        rollouts.append(Rollout(name, rng.random() < accuracy, max(10, int(rng.gauss(length, length * 0.15)))))
    return rollouts


# {{docs-fragment grade}}
@env.task
async def grade(rollouts: list[Rollout], length_bonus: float) -> list[float]:
    """The reward: 1 for a correct answer, plus a bonus per 1,000 tokens of answer."""
    return [float(r.correct) + length_bonus * r.length / 1000 for r in rollouts]
# {{/docs-fragment grade}}


@env.task
async def update(logits: list[float], rollouts: list[Rollout], rewards: list[float], lr: float) -> list[float]:
    """One policy-gradient step with group-normalized advantages (GRPO-style)."""
    mean, std = statistics.mean(rewards), statistics.pstdev(rewards) or 1.0
    advantages = [(r - mean) / std for r in rewards]
    p = _probs(logits)
    grads = [
        sum(a * ((r.behavior == name) - p[k]) for r, a in zip(rollouts, advantages)) / len(rollouts)
        for k, name in enumerate(NAMES)
    ]
    return [x + lr * g for x, g in zip(logits, grads)]


@env.task
async def evaluate(logits: list[float]) -> list[float]:
    """Held-out accuracy and mean answer length under the policy."""
    p = _probs(logits)
    return [
        sum(pi * BEHAVIORS[n][0] for pi, n in zip(p, NAMES)),
        sum(pi * BEHAVIORS[n][1] for pi, n in zip(p, NAMES)),
    ]


# {{docs-fragment train-step}}
@env.task
async def train_step(state: TrainState, gen_latency_s: int) -> StepResult:
    """One step: generate -> grade -> update -> evaluate."""
    rollouts = await generate(state.logits, state.step, state.group_size, gen_latency_s)
    rewards = await grade(rollouts, state.length_bonus)
    logits = await update(state.logits, rollouts, rewards, state.lr)
    accuracy, mean_length = await evaluate(logits)
    next_state = TrainState(state.step + 1, logits, state.lr, state.length_bonus, state.group_size)
    return StepResult(next_state, statistics.mean(rewards), accuracy, mean_length)
# {{/docs-fragment train-step}}


# {{docs-fragment train}}
@env.task(report=True)
async def train(
    steps: int = 12,
    lr: float = 1.0,
    length_bonus: float = 1.0,
    group_size: int = 16,
    gen_latency_s: int = 20,
) -> list[float]:
    """The training loop. Returns held-out accuracy after each step."""
    state = TrainState(0, [1.0, 0.0, -1.0], lr, length_bonus, group_size)
    await flyte.report.log.aio(
        "<style>table{font:14px system-ui;border-collapse:collapse}td,th{padding:4px 14px;text-align:right}"
        "th{border-bottom:1px solid #999}</style><table><tr><th>step</th><th>length bonus</th>"
        "<th>reward</th><th>accuracy</th><th>mean length</th></tr>",
        do_flush=True,
    )
    accuracies = []
    for _ in range(steps):
        result = await train_step(state, gen_latency_s)
        # Read the bonus from the step's result, not from `state`: a fork that edits a step's
        # input changes what that step saw, and the result carries it forward.
        await flyte.report.log.aio(
            f"<tr><td>{state.step}</td><td>{result.state.length_bonus}</td><td>{result.reward:.2f}</td>"
            f"<td>{result.accuracy:.2f}</td><td>{result.mean_length:.0f}</td></tr>",
            do_flush=True,
        )
        accuracies.append(round(result.accuracy, 4))
        state = result.state
    await flyte.report.log.aio("</table>", do_flush=True)
    return accuracies
# {{/docs-fragment train}}


# {{docs-fragment branch}}
async def branch(run_name: str, at_step: int, length_bonus: float = 0.1) -> None:
    """Branch training at `at_step` with a smaller length bonus. The source keeps training."""
    while True:
        async for action in remote.Action.listall.aio(for_run_name=run_name):
            if not (action.task_name or "").endswith("train_step"):
                continue
            details = await remote.ActionDetails.get.aio(run_name=run_name, name=action.name)
            state = (await details.inputs.aio())["state"]
            if state.step == at_step:
                state.length_bonus = length_bonus
                run = await fork.aio(run_name, task_action_inputs={action.name: {"state": state}})
                print(f"Branched {run_name} at step {at_step} -> {run.name}. {run.url}")
                return
        print(f"Waiting for step {at_step} to start...")
        await asyncio.sleep(5)
# {{/docs-fragment branch}}


if __name__ == "__main__":
    flyte.init_from_config()
    if len(sys.argv) >= 4 and sys.argv[1] == "branch":
        asyncio.run(branch(sys.argv[2], int(sys.argv[3])))
    else:
        r = flyte.run(train)
        print(f"Training run: {r.name}\n{r.url}")
        print(
            "\nWhile it trains, branch it:\n"
            f"  python rl_whatif.py branch {r.name} 4      # length_bonus=0.1 from step 4 on\n"
            "or, in the console, open a train_step action -> Edit inputs, lower state.length_bonus,\n"
            "and Fork."
        )
