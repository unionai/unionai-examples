# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "unionai-sandbox==0.0.1b15",
# ]
# main = "eureka"
# params = "generations=2 candidates_per_generation=4"
# ///
"""Eureka-style reward design: Claude writes reward functions, sandboxes train a policy with each.

    for each generation:
        propose_rewards (Claude, traced)  ->  N x evaluate_reward (warm pool, sandboxed PPO)
        -> reflect on the best one and feed it into the next generation
    render_policy (film the winner)  ->  publish the winning reward as an artifact

The task is MountainCar, whose own reward is -1 on every step: PPO trained on it never finds
the flag. Every candidate reward trains a PPO policy on several seeds for a small budget, and
is ranked by how often the greedy policy reaches the flag. The reward code is untrusted, so
each training run happens in a sandbox session: network blocked, filesystem limited to its
working directory, and memory and wall-clock capped.

    python eureka.py              # 4 generations of 6 candidates, about 25 minutes
    python eureka.py --smoke      # 2 generations of 3 candidates
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import flyte
import flyte.artifacts as artifacts
import flyte.report
import prompts
import reports
from flyte.io import File
from union import sandbox as sb

HERE = Path(__file__).parent
MODEL = "claude-opus-5-5"
# The name of the secret that holds your Anthropic API key. Set ANTHROPIC_SECRET when you
# launch to use a secret with another name.
ANTHROPIC_SECRET = os.getenv("ANTHROPIC_SECRET", "anthropic-api-key")

# {{docs-fragment image}}
# The sandbox base image carries the sandbox server, and unionai-reuse the bridge that
# reusable environments need. Torch comes from the CPU-only index.
image = (
    sb.base_image.clone(name="eureka-reward-design")
    .with_pip_packages("torch==2.9.1", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("gymnasium[classic-control]==1.2.3", "numpy<3", "Pillow", "anthropic==2.6.0", "unionai-reuse==0.2.1")
)
# {{/docs-fragment image}}

# {{docs-fragment envs}}
# Candidate evaluation: a warm pool of CPU pods. Each call trains one candidate on every seed
# at once, one sandbox session per seed, and a replica takes one call at a time.
eval_env = flyte.TaskEnvironment(
    name="eureka-eval",
    image=image,
    resources=flyte.Resources(cpu=6, memory="16Gi"),
    reusable=flyte.ReusePolicy(replicas=(1, 6), concurrency=1, idle_ttl=600, scaledown_ttl=180),
    env_vars={"SDL_VIDEODRIVER": "dummy"},
    # The harness is copied into each sandbox, never imported here, so include it explicitly.
    include=["harness.py"],
)

# The driver runs the loop and calls Claude. It holds the API key; the sandboxes never see it.
driver_env = flyte.TaskEnvironment(
    name="eureka-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret(key=ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY")],
    env_vars={"SDL_VIDEODRIVER": "dummy"},
    depends_on=[eval_env],
)
# {{/docs-fragment envs}}


@dataclass
class Candidate:
    name: str
    idea: str
    code: str
    generation: int  # 0 for the baselines


@dataclass
class Result:
    name: str
    idea: str
    code: str
    generation: int
    outcome: str  # "trained", or why the sandbox stopped it
    success: float = 0.0  # task metric, mean over seeds
    per_seed: list[float] = field(default_factory=list)
    steps: float = 200.0  # mean episode length in evaluation; shorter is better
    best_pos: float = -0.5
    seeds: int = 0
    components: dict = field(default_factory=dict)
    curve: list[dict] = field(default_factory=list)
    error: str = ""
    train_s: float = 0.0


def rank(r: Result) -> tuple:
    return (r.success, -r.steps, r.best_pos)


# {{docs-fragment session}}
def python_paths() -> list[str]:
    """Where this image's Python and its packages live, so the sandbox can read them."""
    import sysconfig

    paths = {sys.prefix, sys.base_prefix, sysconfig.get_paths()["purelib"], sysconfig.get_paths()["stdlib"]}
    return sorted(p for p in paths if not p.startswith(("/usr", "/lib", "/bin")))


async def train_in_sandbox(code: str, seed: int, total_steps: int, eval_episodes: int, timeout_s: int) -> dict:
    """Train and grade one policy with one candidate reward, inside a fresh sandbox session."""
    wd = tempfile.mkdtemp(prefix="candidate-")  # the session's only writable path
    config = {"total_steps": total_steps, "num_envs": 16, "seed": seed, "eval_episodes": eval_episodes, "save_policy": True}
    async with sb.on_device.session(
        backend="userns",
        host_work_dir=wd,
        network_mode="blocked",
        read_only_paths=python_paths(),
        timeout_s=timeout_s,
        resources=flyte.Resources(cpu="1", memory="2Gi"),
    ) as session:
        await session.put_bytes(f"{wd}/harness.py", (HERE / "harness.py").read_bytes())
        await session.put_bytes(f"{wd}/reward.py", code.encode())
        await session.put_bytes(f"{wd}/config.json", json.dumps(config).encode())
        proc = await session.run(f"{sys.executable} {wd}/harness.py {wd}", stdout=True, stderr=True)
        out, err = await proc.communicate_text()
    if proc.returncode == 0 and out.strip():
        return {**json.loads(out.strip().splitlines()[-1]), "policy": f"{wd}/policy.pt"}
    return {"error": (err.strip().splitlines() or [f"exit {proc.returncode}"])[-1][:300], "returncode": proc.returncode}
# {{/docs-fragment session}}


def verdict(err: str, elapsed: float, timeout_s: int) -> str:
    if elapsed >= timeout_s - 1:
        return "stopped: wall-clock limit"
    if "PermissionError" in err or "No such file" in err:
        return "stopped: path outside the sandbox"
    if "URLError" in err or "resolution" in err or "Network is unreachable" in err:
        return "stopped: network blocked"
    if "MemoryError" in err:
        return "stopped: memory limit"
    return "failed: raised an error"


# {{docs-fragment evaluate}}
@eval_env.task(retries=1)
async def evaluate_reward(
    candidate: Candidate, total_steps: int = 150_000, seeds: int = 5, eval_episodes: int = 20, timeout_s: int = 420,
) -> tuple[Result, File | None]:
    """Train a policy with one candidate reward on every seed in parallel, and grade it on the task."""
    started = time.time()
    runs = await asyncio.gather(*[
        train_in_sandbox(candidate.code, seed, total_steps, eval_episodes, timeout_s) for seed in range(seeds)
    ])
    elapsed = time.time() - started
    ok = [r for r in runs if "error" not in r]
    base = dict(name=candidate.name, idea=candidate.idea, code=candidate.code, generation=candidate.generation)
    if not ok:
        err = runs[0]["error"]
        print(f"{candidate.name}: {err}")
        return Result(**base, outcome=verdict(err, elapsed, timeout_s), error=err, seeds=seeds), None

    n = len(ok)
    curve = [
        {"step": pts[0]["step"], "train_success": sum(p["train_success"] for p in pts) / n}
        for pts in zip(*[r["curve"] for r in ok])
    ]
    best = max(ok, key=lambda r: (r["success"], -r["steps"]))
    result = Result(
        **base,
        outcome="trained",
        success=sum(r["success"] for r in runs if "error" not in r) / seeds,  # a crashed seed counts as 0
        per_seed=[r.get("success", 0.0) for r in runs],
        steps=sum(r["steps"] for r in ok) / n,
        best_pos=sum(r["best_pos"] for r in ok) / n,
        seeds=seeds,
        components=ok[0]["components"],
        curve=curve,
        error="" if n == seeds else f"{seeds - n} seed(s) failed: {next(r['error'] for r in runs if 'error' in r)}",
        train_s=round(elapsed, 1),
    )
    print(f"{candidate.name}: success {result.success:.0%} per seed {result.per_seed} in {elapsed:.0f}s")
    return result, await File.from_local(best["policy"])
# {{/docs-fragment evaluate}}


# {{docs-fragment propose}}
@flyte.trace
async def propose_rewards(generation: int, num_candidates: int, history_json: str) -> str:
    """Ask Claude for reward functions. Traced, so a retried driver reuses the answer instead of paying again."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    async with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32_000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=prompts.SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": prompts.CANDIDATES_SCHEMA}},
        messages=[{"role": "user", "content": prompts.proposal_prompt(num_candidates, json.loads(history_json))}],
    ) as stream:
        message = await stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to write rewards for generation {generation}")
    text = next(b.text for b in message.content if b.type == "text")
    print(f"generation {generation}: {message.usage.input_tokens} input, {message.usage.output_tokens} output tokens")
    return text
# {{/docs-fragment propose}}


@eval_env.task
async def render_policy(policy: File, frames: int = 200) -> str:
    """Film one greedy episode of a trained policy and return it as a base64 GIF."""
    import base64
    import io

    import gymnasium as gym
    import harness
    import torch
    from PIL import Image

    net = harness.ActorCritic()
    net.load_state_dict(torch.load(await policy.download(), weights_only=True))
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=12345)
    shots = [env.render()]
    for _ in range(frames):
        with torch.no_grad():
            logits, _ = net(harness.scale(obs[None]))
        obs, _, terminated, truncated, _ = env.step(int(logits.argmax()))
        shots.append(env.render())
        if terminated or truncated:
            break
    images = [Image.fromarray(s).resize((300, 200)) for s in shots[::2]]
    buf = io.BytesIO()
    images[0].save(buf, "GIF", save_all=True, append_images=images[1:], duration=40, loop=0)
    return base64.b64encode(buf.getvalue()).decode()


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True)
async def eureka(
    generations: int = 4,
    candidates_per_generation: int = 6,
    total_steps: int = 150_000,
    seeds: int = 5,
    include_hostile: bool = True,
) -> File:
    """Evolve reward functions for MountainCar with Claude, and publish the best one."""
    budget = dict(total_steps=total_steps, seeds=seeds)
    baselines = asyncio.gather(
        evaluate_reward(Candidate("environment_reward", "the environment's own -1 per step", prompts.SPARSE, 0), **budget),
        evaluate_reward(Candidate("human_energy_shaping", "mechanical energy gain plus a goal bonus", prompts.HUMAN, 0), **budget),
    )
    results: list[Result] = []
    policies: dict[str, File] = {}
    history: list[dict] = []  # the best of each generation, which is what Claude reflects on

    for g in range(1, generations + 1):
        with flyte.group(f"generation-{g}"):
            text = await propose_rewards(g, candidates_per_generation, json.dumps(history))
            candidates = [Candidate(c["name"], c["idea"], c["code"], g) for c in json.loads(text)["candidates"]]
            if g == 1 and include_hostile:
                candidates += [Candidate(n, "hostile code, to show the sandbox", c, g) for n, c in prompts.HOSTILE.items()]
            evaluated = await asyncio.gather(*[evaluate_reward(c, **budget) for c in candidates])
        for r, policy in evaluated:
            results.append(r)
            if policy:
                policies[f"{r.generation}/{r.name}"] = policy
        best = max((r for r, _ in evaluated), key=rank)
        history.append({**asdict(best), "seeds": seeds,
                        "scoreboard": [(r.name, r.idea, r.success, r.outcome) for r, _ in evaluated
                                       if r.name not in prompts.HOSTILE]})
        await flyte.report.replace.aio(reports.live(results, history, None, generations), do_flush=True)

    base = [r for r, _ in await baselines]
    winner = max(results, key=rank)
    gif = await render_policy(policies[f"{winner.generation}/{winner.name}"]) if winner.success > 0 else ""
    await flyte.report.replace.aio(reports.final(results, history, base, winner, gif), do_flush=True)
    flyte.report.get_tab("Candidates").replace(reports.candidates_tab(results))
    await flyte.report.flush.aio()

    path = os.path.join(tempfile.mkdtemp(), f"{winner.name}.py")
    Path(path).write_text(winner.code)
    card = await artifacts.Card.create_from.aio(content=reports.card(winner, base, history, gif), format="html", card_type="model")
    metadata = artifacts.Metadata(
        name="mountaincar-reward-function",
        description=f"Best of {len(results)} Claude-written MountainCar rewards: flag reached in {winner.success:.0%} of episodes",
        kind="model",
        attrs={"candidate": winner.name, "generation": str(winner.generation), "success": f"{winner.success:.3f}",
               "total_steps": str(total_steps), "seeds": str(seeds), "model": MODEL},
        card=card,
    )
    return artifacts.new(await File.from_local(path), metadata)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    if "--smoke" in sys.argv:
        run = flyte.run(eureka, generations=2, candidates_per_generation=3)
    else:
        run = flyte.run(eureka)
    print(run.url)
# {{/docs-fragment run}}
