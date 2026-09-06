"""Durable agentic RL on OSS Flyte 2 — the open-source counterpart to `agentic_rl_durable.py`.

Same SkyRL/Harbor-style shape: a policy attempts tasks inside per-trial sandboxes, a verifier +
LLM judge score each attempt, a trainer updates the policy, and the loop repeats. The training is
the same toy bandit, and the durability core is identical:

  [L3]  Every trial is TWO Flyte actions: `generate` (the sandbox rollout) and
        `verify_and_judge` (rubric is an explicit input). Completed trials survive a driver
        crash; a failed trial is retried or skipped, never fatal to the step.
  [J]   The judge call is `@flyte.trace`d: a retry of the judging task replays it instead of
        paying for it again.
  [L2b] Policy weights are small and flow trainer -> rollout as an *input* (pull-based sync).

Where the Union-backend version uses Union-only features, this file uses OSS equivalents:

  - Worlds are cached task outputs of type `flyte.io.Dir` (downloaded per trial) instead of
    `Volume.fork()` copies. Slower per trial for big worlds, but the isolation story is the
    same: each trial's pod gets its own private copy, and writes never leak.
  - The trainer is a pure, deterministic `train_step` task that threads weights through as
    inputs/outputs (a checkpoint per step) instead of holding state in a detached Ray actor on
    a reusable cluster. Exactly-once comes free: a replayed step recomputes identical outputs.
  - Re-scoring with a new rubric is a *new run* that leans on caching (`cache="auto"` on
    `build_world` and `generate`) instead of `flyte.rerun(recover=True, ...)`. Worlds and any
    rollouts whose inputs are unchanged are reused; judging and training re-run.

Run (CPU):
    flyte --config <your-config> run agentic_rl_durable_oss.py train
Re-score with a new rubric (caching reuses worlds + unchanged rollouts):
    flyte --config <your-config> run agentic_rl_durable_oss.py rescore
Crash-injection knobs: --crash_driver_at_step 1 --flaky_trial_rate 0.25 --judge_flake_rate 0.3
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

import flyte
import flyte.durable
import flyte.errors
import flyte.report
from flyte.io import Dir

# {{docs-fragment image}}
# One small CPU image; no Ray, no FUSE, no Union plugins.
image = flyte.Image.from_debian_base(name="agentic-rl-durable-oss").with_pip_packages("pydantic")
# {{/docs-fragment image}}

# --- environments ------------------------------------------------------------------------------

# {{docs-fragment envs}}
# Separate environments so each tier's resources, retries, and timeouts are its own.
sandbox_env = flyte.TaskEnvironment(
    name="agentic-rl-oss-sandbox",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)

judge_env = flyte.TaskEnvironment(
    name="agentic-rl-oss-judge", image=image, resources=flyte.Resources(cpu="500m", memory="512Mi")
)

trainer_env = flyte.TaskEnvironment(
    name="agentic-rl-oss-trainer", image=image, resources=flyte.Resources(cpu="500m", memory="512Mi")
)

driver_env = flyte.TaskEnvironment(
    name="agentic-rl-oss-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[sandbox_env, judge_env, trainer_env],
)
# {{/docs-fragment envs}}

# --- types ---------------------------------------------------------------------------------------


# {{docs-fragment types}}
class Weights(BaseModel):
    """[L2b] The whole 'policy' — small enough to travel as an input to every rollout."""

    version: int = 0
    # per world: prior over which document holds the secret (the bandit's arm weights)
    doc_prior: dict[str, list[float]] = {}


class TrialSpec(BaseModel):
    trial_id: str
    step: int
    world_id: str
    prompt: str
    max_turns: int = 6


class Trajectory(BaseModel):
    trial_id: str
    world_id: str
    weights_version: int
    turns: list[dict]  # [{doc, found}]
    answer: Optional[str]
    files_written: list[str]
    elapsed_s: float


class Rubric(BaseModel):
    """Judge configuration is an INPUT, so changing it changes only judging + training."""

    correctness_weight: float = 1.0
    efficiency_weight: float = 0.3
    tidiness_weight: float = 0.1
    name: str = "v1"


class Reward(BaseModel):
    trial_id: str
    world_id: str
    verified_correct: bool
    n_turns: int
    judge_score: float
    reward: float
    found_doc: Optional[int]


class StepResult(BaseModel):
    step: int
    weights: Weights
    mean_reward: float
    mean_turns: float
    n_trials: int


# {{/docs-fragment types}}

# --- helpers -------------------------------------------------------------------------------------


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _world_secret(world_id: str) -> tuple[int, str]:
    r = random.Random(_seed("world", world_id))
    return r.randrange(N_DOCS), f"{world_id}-secret-{r.randrange(10**6):06d}"


def _world_secret_digest(world_id: str) -> str:
    """Only the digest is ever written to disk or compared — the raw secret never leaves this module."""
    _, secret = _world_secret(world_id)
    return hashlib.sha256(secret.encode()).hexdigest()


N_DOCS = 8


# --- worlds: built once (cached), downloaded per trial -------------------------------------------


# {{docs-fragment build-world}}
@sandbox_env.task(cache="auto")
async def build_world(world_id: str) -> Dir:
    """Populate the world's filesystem once; uploaded as a Dir every trial downloads.

    Cached, so every run (including a re-scoring run) reuses the same Dir — which is what makes
    downstream `generate` cache keys line up across runs.
    """
    secret_doc, secret = _world_secret(world_id)
    root = Path(tempfile.mkdtemp(prefix=f"world-{world_id}-")) / "docs"
    root.mkdir(parents=True)
    for i in range(N_DOCS):
        lines = [f"title: report {i} for {world_id}", f"owner: team-{i % 3}", f"pages: {10 + i}"]
        if i == secret_doc:
            lines.append(f"secret_sha256: {hashlib.sha256(secret.encode()).hexdigest()}")
        (root / f"doc_{i}.txt").write_text("\n".join(lines) + "\n")
    return await Dir.from_local(root)


# {{/docs-fragment build-world}}

# --- [L3] trial action 1: generate (the sandbox rollout) -----------------------------------------


# {{docs-fragment generate}}
@sandbox_env.task(cache="auto", retries=1, timeout=flyte.Timeout(max_runtime=300))
async def generate(spec: TrialSpec, weights: Weights, world: Dir, flaky_rate: float = 0.0) -> Trajectory:
    """One trial: download a private copy of the world, run the agent loop, return what happened.

    Idempotent by construction — re-running produces the same trajectory (seeded by trial_id),
    so a retry is a clean regeneration. Cached, so a re-scoring run with identical inputs
    (same spec, same weights, same world) reuses the rollout instead of regenerating it.
    The pod's own filesystem is the sandbox: writes are private and vanish with the pod.
    """
    t0 = time.monotonic()
    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    rng = random.Random(_seed("trial", spec.trial_id))
    if attempt == 0 and rng.random() < flaky_rate:
        raise RuntimeError(f"simulated sandbox failure for {spec.trial_id} (attempt 0)")  # -> retried

    # private copy of the world for this trial (downloaded into this pod)
    ws = Path(await world.download())
    docs = ws if (ws / "doc_0.txt").exists() else ws / "docs"

    # the "agent": pick documents to read in an order driven by the policy's prior for this world
    prior = list(weights.doc_prior.get(spec.world_id) or [1.0] * N_DOCS)
    order: list[int] = []
    remaining = list(range(N_DOCS))
    while remaining:
        w = [prior[i] for i in remaining]
        pick = rng.choices(remaining, weights=w, k=1)[0]
        order.append(pick)
        remaining.remove(pick)

    turns, answer = [], None
    for doc in order[: spec.max_turns]:
        text = (docs / f"doc_{doc}.txt").read_text()
        found = "secret_sha256:" in text
        turns.append({"doc": doc, "found": found})
        await asyncio.sleep(0.2)  # a tool call
        if found:
            answer = text.split("secret_sha256:")[1].strip()
            break

    written = []
    (docs.parent / "answer.txt").write_text(answer or "")
    written.append("answer.txt")
    if rng.random() < 0.3:  # untidy agents leave scratch files behind (the judge cares)
        (docs.parent / "scratch.tmp").write_text("notes")
        written.append("scratch.tmp")

    return Trajectory(
        trial_id=spec.trial_id,
        world_id=spec.world_id,
        weights_version=weights.version,
        turns=turns,
        answer=answer,
        files_written=written,
        elapsed_s=round(time.monotonic() - t0, 2),
    )


# {{/docs-fragment generate}}

# --- [L3]+[J] trial action 2: verify + judge (rubric is an input) --------------------------------


# {{docs-fragment judge}}
@flyte.trace
async def call_judge(traj: Trajectory, rubric: Rubric) -> float:
    """Stand-in for the vendor LLM judge. Traced: replayed, not re-called, on a task retry."""
    print(f"JUDGE CALLED trial={traj.trial_id} rubric={rubric.name}", flush=True)  # count these in logs
    await asyncio.sleep(0.5)  # network
    rng = random.Random(_seed("judge", traj.trial_id, rubric.name))
    correct = 1.0 if traj.answer == _world_secret_digest(traj.world_id) else 0.0
    efficiency = max(0.0, 1.0 - (len(traj.turns) - 1) / N_DOCS)
    tidy = 1.0 if traj.files_written == ["answer.txt"] else 0.0
    score = (
        rubric.correctness_weight * correct + rubric.efficiency_weight * efficiency + rubric.tidiness_weight * tidy
    ) + rng.uniform(-0.05, 0.05)
    return round(score, 3)


@judge_env.task(retries=2)
async def verify_and_judge(traj: Trajectory, rubric: Rubric, judge_flake_rate: float = 0.0) -> Reward:
    verified = traj.answer == _world_secret_digest(traj.world_id)  # the deterministic verifier
    score = await call_judge(traj, rubric)  # [J] memoized across retries

    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    if attempt == 0 and random.Random(_seed("flake", traj.trial_id)).random() < judge_flake_rate:
        raise RuntimeError("simulated post-judge failure (e.g. upload) — retry must NOT re-call the judge")

    found_doc = next((t["doc"] for t in traj.turns if t["found"]), None)
    return Reward(
        trial_id=traj.trial_id,
        world_id=traj.world_id,
        verified_correct=verified,
        n_turns=len(traj.turns),
        judge_score=score,
        reward=round((1.0 if verified else 0.0) + score, 3),
        found_doc=found_doc,
    )


# {{/docs-fragment judge}}

# --- trainer: a pure step function; weights are the checkpoint ------------------------------------


# {{docs-fragment train-step}}
@trainer_env.task(retries=1, timeout=flyte.Timeout(max_runtime=600))
async def train_step(step: int, weights: Weights, rewards: list[Reward], lr: float = 2.0) -> StepResult:
    """One optimizer step as a pure function: (weights in, rewards in) -> new weights out.

    The returned weights ARE the checkpoint — every step's output is durably recorded, so a
    driver crash resumes from the last completed step. Determinism makes re-execution safe:
    a replayed step recomputes the identical update, so there is no double-apply to guard
    against (the stateful-actor version needs an explicit exactly-once check instead).
    """
    new = weights.model_copy(deep=True)
    for r in rewards:
        if r.found_doc is not None and r.reward > 0:
            prior = new.doc_prior[r.world_id]
            prior[r.found_doc] += lr * r.reward  # bandit update toward the rewarded doc
    new.version += 1
    return StepResult(
        step=step,
        weights=new,
        mean_reward=round(sum(r.reward for r in rewards) / max(1, len(rewards)), 3),
        mean_turns=round(sum(r.n_turns for r in rewards) / max(1, len(rewards)), 2),
        n_trials=len(rewards),
    )


# {{/docs-fragment train-step}}

# --- driver ---------------------------------------------------------------------------------------


# {{docs-fragment driver}}
@driver_env.task(report=True)
async def train(
    n_steps: int = 3,
    n_worlds: int = 3,
    prompts_per_step: int = 3,
    n_samples: int = 2,
    rubric: Rubric = Rubric(),
    crash_driver_at_step: int = -1,
    flaky_trial_rate: float = 0.0,
    judge_flake_rate: float = 0.0,
) -> list[StepResult]:
    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    started = flyte.durable.now()  # recorded once; replayed on retry
    world_ids = [f"w{i}" for i in range(n_worlds)]

    # worlds are cached; initial weights are deterministic, so step-0 generate inputs
    # hash identically across runs (which is what lets a re-scoring run reuse them)
    worlds = dict(zip(world_ids, await asyncio.gather(*[build_world(w) for w in world_ids])))
    weights = Weights(version=0, doc_prior={w: [1.0] * N_DOCS for w in world_ids})

    history: list[StepResult] = []
    tab = flyte.report.get_tab("training")
    for step in range(n_steps):
        with flyte.group(f"step-{step}"):
            # [L3] fan out trials; [L2b] every trial carries the current weights as an input
            specs = [
                TrialSpec(
                    trial_id=f"s{step}-p{p}-n{s}", step=step, world_id=world_ids[p % n_worlds], prompt="find the secret"
                )
                for p in range(prompts_per_step)
                for s in range(n_samples)
            ]
            gens = [asyncio.create_task(generate(sp, weights, worlds[sp.world_id], flaky_trial_rate)) for sp in specs]

            # pipeline: judge each trajectory the moment it lands; a failed trial is skipped, not fatal
            judges = []
            for fut in asyncio.as_completed(gens):
                try:
                    traj = await fut
                except Exception as e:  # noqa: BLE001 — skip_failed_rollouts
                    print(f"trial failed after retries, skipping: {e}", flush=True)
                    continue
                judges.append(asyncio.create_task(verify_and_judge(traj, rubric, judge_flake_rate)))
            results = await asyncio.gather(*judges, return_exceptions=True)
            # canonical order: as_completed order differs between attempts, and the step's inputs
            # must hash identically for the replayed step to be reused
            rewards = sorted((r for r in results if isinstance(r, Reward)), key=lambda r: r.trial_id)

            # one durable step; its output weights are the checkpoint
            result = await train_step(step, weights, rewards)
            weights = result.weights
            history.append(result)

            tab.log(
                f"<p><b>step {step}</b> · trials {result.n_trials}/{len(specs)} · mean reward "
                f"{result.mean_reward} · mean turns {result.mean_turns} · weights v{weights.version} · "
                f"driver attempt {attempt}</p>"
            )
            await flyte.report.flush.aio()

            if attempt == 0 and step == crash_driver_at_step:
                raise flyte.errors.RuntimeSystemError("simulated", f"driver crash after step {step} on attempt 0")

    tab.log(f"<p>started {started.isoformat()} · finished on attempt {attempt}</p>")
    await flyte.report.flush.aio()
    return history


# {{/docs-fragment driver}}

# --- re-score: a new run with a new rubric; caching finds the reuse frontier ----------------------


# {{docs-fragment rescore}}
@driver_env.task(report=True)
async def rescore(
    n_steps: int = 3,
    n_worlds: int = 3,
    prompts_per_step: int = 3,
    n_samples: int = 2,
) -> list[StepResult]:
    """Re-run training with a new rubric. Cached `build_world` and `generate` actions are reused
    wherever their inputs are unchanged; judging and training re-run everywhere.

    Step 0 reuses every rollout (same worlds, same v0 weights). The new rewards change the
    weights after step 0, so later steps' `generate` inputs differ and regenerate — cache-key
    identity finds that invalidation frontier by itself; nothing here knows about "forks".
    """
    new = Rubric(correctness_weight=1.0, efficiency_weight=1.0, tidiness_weight=0.5, name="v2-efficiency")
    return await train(
        n_steps=n_steps, n_worlds=n_worlds, prompts_per_step=prompts_per_step, n_samples=n_samples, rubric=new
    )


# {{/docs-fragment rescore}}

if __name__ == "__main__":
    import sys

    flyte.init_from_config(os.environ.get("FLYTE_CONFIG"))
    r = flyte.run(rescore if len(sys.argv) > 1 and sys.argv[1] == "rescore" else train)
    print(r.url)
