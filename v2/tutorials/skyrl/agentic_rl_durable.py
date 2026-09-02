"""Agentic RL on Flyte — every idea from design.md in one runnable, CPU-only example.

The *shape* is SkyRL/Harbor-style agentic RL: a policy attempts tasks inside
per-trial sandboxes, a verifier + LLM judge score each attempt, a trainer updates the policy, and
the loop repeats. The *training* here is a toy bandit so it runs on CPU — but every structural
decision is the real one:

  [L1+] The trainer lives in a detached, namespaced Ray actor on ONE reusable Ray cluster, and
        every training step is its own Flyte task -> the step is a durable boundary; a driver
        crash replays completed steps and resumes on the same warm cluster.
  [L3]  Every trial is TWO Flyte actions: `generate` (the sandbox rollout) and
        `verify_and_judge` (rubric is an explicit input). Completed trials survive a driver
        crash; a failed trial is retried or skipped, never fatal to the step.
  [S2]  The sandbox is the task pod; the world filesystem is a `Volume.fork()` per trial
        (< 1 s regardless of world size, isolated writes, resettable for free).
  [L2b] Policy weights are small and flow trainer -> rollout as an *input* (pull-based sync),
        so the rollout tier needs no NCCL and no engine control plane.
  [J]   The judge call is `@flyte.trace`d: a retry of the judging task replays it instead of
        paying for it again. (Keep your vendor judge; just never re-bill.)
  [F]   Fork: `flyte.rerun(run, recover=True, rubric=<new>)` reuses every `generate` action and
        re-runs only `verify_and_judge` + `train_step` — re-score old rollouts with a new rubric
        for judge cost only.

Run (CPU, org-demo):
    flyte --config ~/.flyte/demo-ketan.yaml run agentic_rl_durable.py train
Fork with a new rubric (re-scores without regenerating):
    python agentic_rl_durable.py fork <run-name>
Crash-injection knobs: --crash_driver_at_step 1 --flaky_trial_rate 0.25 --judge_flake_rate 0.3
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from flyteplugins.union.io import ROVolume, Volume
from pydantic import BaseModel

import flyte
import flyte.errors
import flyte.report
import flyte.durable

try:  # local parse without the plugin installed
    from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(),
        worker_node_config=[WorkerNodeConfig(group_name="w", replicas=1)],
        enable_autoscaling=False,
    )
except ImportError:
    ray_config = None

RAY_NAMESPACE = "agentic-rl"
POLICY_ACTOR = "policy-trainer"

# One image for everything: Ray (+ wget for the reusable head's probe), Volumes, local SDK on top.
image = (
    flyte.Image.from_debian_base(install_flyte=False, name="agentic-rl-durable")
    .with_apt_packages("wget")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray", "flyteplugins-union>=0.8.2", "pydantic")
    .with_local_v2()
)

# --- environments ------------------------------------------------------------------------------

# [L1+] one long-lived Ray cluster for the whole run; each step is a task on it.
trainer_env = flyte.TaskEnvironment(
    name="agentic-rl-trainer",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("2000Mi", "4000Mi")),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600, scope="run"),
)

# [S2] pod-as-sandbox: FUSE for the forked world volume; one fresh pod per trial.
sandbox_env = flyte.TaskEnvironment(
    name="agentic-rl-sandbox",
    image=image,
    pod_template=flyte.PodTemplate().allow_fuse(),
    resources=flyte.Resources(cpu="500m", memory="1Gi"),
)

# [J] judging is cheap CPU; separate env so its retries/timeouts are its own.
judge_env = flyte.TaskEnvironment(
    name="agentic-rl-judge", image=image, resources=flyte.Resources(cpu="500m", memory="512Mi")
)

driver_env = flyte.TaskEnvironment(
    name="agentic-rl-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[trainer_env, sandbox_env, judge_env],
)

# --- types ---------------------------------------------------------------------------------------


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
    fork_name: str


class Rubric(BaseModel):
    """[F] Judge configuration is an INPUT, so changing it changes only the judging actions."""

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
    actor_pid: int


# --- helpers -------------------------------------------------------------------------------------


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _world_secret(world_id: str) -> tuple[int, str]:
    r = random.Random(_seed("world", world_id))
    return r.randrange(N_DOCS), f"{world_id}-secret-{r.randrange(10**6):06d}"


N_DOCS = 8


# --- [S2] worlds: built once (cached), forked per trial ---------------------------------------------


@sandbox_env.task(cache="auto")
async def build_world(world_id: str) -> ROVolume:
    """Populate the world's filesystem once; sealed as a read-only volume every trial forks."""
    secret_doc, secret = _world_secret(world_id)
    vol = Volume.new(name=f"world-{world_id}-{flyte.ctx().action.run_name}")
    await vol.mount()
    root = Path(vol.mount_path) / "docs"
    root.mkdir(parents=True)
    for i in range(N_DOCS):
        lines = [f"title: report {i} for {world_id}", f"owner: team-{i % 3}", f"pages: {10 + i}"]
        if i == secret_doc:
            lines.append(f"secret_sha256: {hashlib.sha256(secret.encode()).hexdigest()}")
        (root / f"doc_{i}.txt").write_text("\n".join(lines) + "\n")
    return await vol.finalize(message=f"world {world_id}")


# --- [L3] trial action 1: generate (the sandbox rollout) -----------------------------------------------


@sandbox_env.task(retries=1, timeout=flyte.Timeout(max_runtime=300))
async def generate(spec: TrialSpec, weights: Weights, world: ROVolume, flaky_rate: float = 0.0) -> Trajectory:
    """One trial: fork the world, run the agent loop against it, return what happened.

    Idempotent by construction — re-running produces the same trajectory (seeded by trial_id),
    so a retry is a clean regeneration, not a replay of stale sandbox state.
    """
    t0 = time.monotonic()
    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    rng = random.Random(_seed("trial", spec.trial_id))
    if attempt == 0 and rng.random() < flaky_rate:
        raise RuntimeError(f"simulated sandbox failure for {spec.trial_id} (attempt 0)")  # -> retried

    # [S2] instant private copy of the world for this trial
    fork_name = f"{world.name}-{spec.trial_id}"
    ws = await world.fork(name=fork_name)
    await ws.mount()
    docs = Path(ws.mount_path) / "docs"

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
        found = "secret:" in text
        turns.append({"doc": doc, "found": found})
        await asyncio.sleep(0.2)  # a tool call
        if found:
            answer = text.split("secret:")[1].strip()
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
        fork_name=fork_name,
    )


# --- [L3]+[J] trial action 2: verify + judge (rubric is an input) ------------------------------------


@flyte.trace
async def call_judge(traj: Trajectory, rubric: Rubric) -> float:
    """Stand-in for the vendor LLM judge. Traced: replayed, not re-called, on a task retry."""
    print(f"JUDGE CALLED trial={traj.trial_id} rubric={rubric.name}", flush=True)  # count these in logs
    await asyncio.sleep(0.5)  # network
    rng = random.Random(_seed("judge", traj.trial_id, rubric.name))
    _, secret = _world_secret(traj.world_id)
    correct = 1.0 if traj.answer == secret else 0.0
    efficiency = max(0.0, 1.0 - (len(traj.turns) - 1) / N_DOCS)
    tidy = 1.0 if traj.files_written == ["answer.txt"] else 0.0
    score = (
        rubric.correctness_weight * correct + rubric.efficiency_weight * efficiency + rubric.tidiness_weight * tidy
    ) + rng.uniform(-0.05, 0.05)
    return round(score, 3)


@judge_env.task(retries=2)
async def verify_and_judge(traj: Trajectory, rubric: Rubric, judge_flake_rate: float = 0.0) -> Reward:
    _, secret = _world_secret(traj.world_id)
    verified = traj.answer == secret  # the deterministic verifier (tests/test.sh in Harbor)
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


# --- [L1+] trainer: detached actor on the reusable cluster; one task per step --------------------------


def _policy_actor():
    import ray

    @ray.remote
    class PolicyTrainer:
        """Holds the trainable state (think: FSDP shards + optimizer) across steps."""

        def __init__(self, world_ids: list[str]):
            self.weights = Weights(version=0, doc_prior={w: [1.0] * N_DOCS for w in world_ids})
            self.pid = os.getpid()
            self.applied: dict[int, dict] = {}  # step -> weights after that step (exactly-once)

        def update(self, step: int, rewards: list[dict], lr: float) -> dict:
            # A re-executed step task (retry after the actor already applied it, or a driver
            # replay with a differently-ordered batch) must not apply the update twice.
            if step in self.applied:
                return self.applied[step]
            for r in rewards:
                if r["found_doc"] is not None and r["reward"] > 0:
                    prior = self.weights.doc_prior[r["world_id"]]
                    prior[r["found_doc"]] += lr * r["reward"]  # bandit update toward the rewarded doc
            self.weights.version += 1
            self.applied[step] = self.weights.model_dump()
            return self.applied[step]

        def info(self) -> dict:
            return {"pid": self.pid, "version": self.weights.version}

    return PolicyTrainer


def _attach_policy(world_ids: list[str]):
    import ray

    try:
        return ray.get_actor(POLICY_ACTOR, namespace=RAY_NAMESPACE)
    except ValueError:
        return (
            _policy_actor()
            .options(name=POLICY_ACTOR, namespace=RAY_NAMESPACE, lifetime="detached", get_if_exists=True)
            .remote(world_ids)
        )


@trainer_env.task
async def setup_trainer(world_ids: list[str]) -> Weights:
    import ray

    actor = _attach_policy(world_ids)
    info = ray.get(actor.info.remote())
    print(f"trainer actor pid={info['pid']} version={info['version']}", flush=True)
    return Weights(version=info["version"], doc_prior={w: [1.0] * N_DOCS for w in world_ids})


@trainer_env.task(retries=1, timeout=flyte.Timeout(max_runtime=600))
async def train_step(step: int, rewards: list[Reward], world_ids: list[str], lr: float = 2.0) -> StepResult:
    """One optimizer step. Re-attaches to the warm actor; returns the new (small) weights."""
    import ray

    actor = _attach_policy(world_ids)
    new_weights = ray.get(actor.update.remote(step, [r.model_dump() for r in rewards], lr))
    info = ray.get(actor.info.remote())
    return StepResult(
        step=step,
        weights=Weights(**new_weights),
        mean_reward=round(sum(r.reward for r in rewards) / max(1, len(rewards)), 3),
        mean_turns=round(sum(r.n_turns for r in rewards) / max(1, len(rewards)), 2),
        n_trials=len(rewards),
        actor_pid=info["pid"],
    )


# --- driver ----------------------------------------------------------------------------------------


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

    # worlds are cached; the trainer actor is created once per run and re-attached afterwards
    worlds = dict(zip(world_ids, await asyncio.gather(*[build_world(w) for w in world_ids])))
    weights = await setup_trainer(world_ids)

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
            # must hash identically for the replayed step to be reused (lesson from run upq5qwpv)
            rewards = sorted((r for r in results if isinstance(r, Reward)), key=lambda r: r.trial_id)

            # [L1+] one durable step on the warm cluster
            result = await train_step(step, rewards, world_ids)
            weights = result.weights
            history.append(result)

            tab.log(
                f"<p><b>step {step}</b> · trials {result.n_trials}/{len(specs)} · mean reward "
                f"{result.mean_reward} · mean turns {result.mean_turns} · weights v{weights.version} · "
                f"actor pid {result.actor_pid} · driver attempt {attempt}</p>"
            )
            await flyte.report.flush.aio()

            if attempt == 0 and step == crash_driver_at_step:
                raise flyte.errors.RuntimeSystemError("simulated", f"driver crash after step {step} on attempt 0")

    tab.log(f"<p>started {started.isoformat()} · finished on attempt {attempt}</p>")
    await flyte.report.flush.aio()
    return history


# --- [F] fork: re-score the same rollouts with a new rubric ----------------------------------------------


def fork_with_new_rubric(run_name: str) -> None:
    """Every `generate` is reused; `verify_and_judge` (new rubric input) and `train_step` re-run."""
    new = Rubric(correctness_weight=1.0, efficiency_weight=1.0, tidiness_weight=0.5, name="v2-efficiency")
    r = flyte.rerun(run_name, recover=True, rubric=new)
    print("forked run:", r.url)


if __name__ == "__main__":
    import sys

    flyte.init_from_config(os.environ.get("FLYTE_CONFIG"))  # e.g. FLYTE_CONFIG=~/.flyte/demo.yaml
    if len(sys.argv) > 2 and sys.argv[1] == "fork":
        fork_with_new_rubric(sys.argv[2])
    else:
        r = flyte.run(train)
        print(r.url)
