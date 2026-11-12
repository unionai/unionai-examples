# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "flyte>=2.13.0",
#    "anthropic>=0.70",
#    "pydantic>=2",
#    "reportlab", "openpyxl", "python-pptx", "pypdf", "pdfplumber",
# ]
# main = "compare_harnesses"
# params = ""
# ///
"""Harness-first evals for agentic RL: the harness is an input, every (task × harness × model × pass)
cell is a cached action, and adding a harness version or a model pays only for the new cells.

Mercor's 397B SkyRL guide reports that harness fixes alone raised its untrained 35B from 22.74% to 28.69%
mean reward. This example makes that loop cheap and repeatable:

- ``build_world``      a company's shared drive (PDF, XLSX, PPTX, inbox), built once per world (cached).
- ``run_trial``        one agent attempt (Claude, tool use) on a private copy of the world, under a harness.
                       Cached on (task, policy, harness, world, pass). Timeouts and retries are task config.
- ``grade``            deterministic verifier + a traced LLM judge. Split from ``run_trial`` so a new rubric
                       re-grades stored trajectories without re-rolling them.
- ``evaluate``         fans out the cells for one (policy, harness) and summarizes them.
- ``compare_harnesses`` v1 vs v2 on the same model, plus an LLM audit of v1's failures that proposes the bugs.
- ``leaderboard``      several models on one harness; models already evaluated come back from cache.

Run::

    flyte run harness_eval.py compare_harnesses
    flyte run harness_eval.py leaderboard --policies '[{"model": "claude-haiku-5-5"}, {"model": "claude-sonnet-5-5"}]'
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import flyte
import flyte.errors
import flyte.io
import flyte.report
from harness_world import (
    HARNESS_V1,
    HARNESS_V2,
    Harness,
    Policy,
    TaskSpec,
    Trajectory,
    World,
    build_world_files,
    make_tasks,
    run_agent,
    verify,
)
from pydantic import BaseModel

# {{docs-fragment envs}}
image = flyte.Image.from_debian_base(name="agentic-rl-harness-eval", python_version=(3, 12)).with_pip_packages(
    "anthropic>=0.70", "pydantic>=2", "reportlab", "openpyxl", "python-pptx", "pypdf", "pdfplumber",
    "unionai-reuse>=0.1.3",
)
# The secret holding an Anthropic API key. Override with ANTHROPIC_SECRET=<name> when you launch; it's passed into
# the containers too, so the module resolves the same secret name locally and remotely.
ANTHROPIC_SECRET = os.environ.get("ANTHROPIC_SECRET", "ANTHROPIC_API_KEY")
ANTHROPIC = flyte.Secret(key=ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY")
SECRET_ENV = {"ANTHROPIC_SECRET": ANTHROPIC_SECRET}

world_env = flyte.TaskEnvironment(name="harness-eval-world", image=image, resources=flyte.Resources(cpu=1, memory="1Gi"))

# Trials are I/O-bound (model API calls), so a warm pool runs several per replica. Each trial still gets its
# own copy of the world, its own subprocess per code call, and its own deadline. For MCP-server-in-the-sandbox
# worlds, where agent loops sharing a process disconnect each other, use .override(reusable="off") for a pod
# per trial.
trial_env = flyte.TaskEnvironment(
    # Warm replicas keep the secrets they started with. If you rotate the key, give the pool a new name (or let
    # it idle out) so trials don't keep running with the old one.
    name="harness-eval-trial-pool",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    reusable=flyte.ReusePolicy(replicas=(1, 6), concurrency=8, idle_ttl=300, scaledown_ttl=120),
    secrets=[ANTHROPIC],
    env_vars=SECRET_ENV,
)
grade_env = flyte.TaskEnvironment(
    name="harness-eval-grade", image=image, resources=flyte.Resources(cpu=1, memory="1Gi"), secrets=[ANTHROPIC],
    env_vars=SECRET_ENV,
)
driver_env = flyte.TaskEnvironment(
    name="harness-eval-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[ANTHROPIC],
    env_vars=SECRET_ENV,
    depends_on=[world_env, trial_env, grade_env],
)
# {{/docs-fragment envs}}

# Model API transients (429s past the SDK's own retries, 5xx, dropped connections) retry the trial with backoff.
API_RETRIES = flyte.RetryStrategy(
    count=3, backoff=flyte.Backoff(base=timedelta(seconds=15), factor=2.0, cap=timedelta(minutes=2))
)


class Rubric(BaseModel):
    version: str = "r1"
    criteria: list[str] = [
        "The justification names the file(s) the answer came from.",
        "The justification shows how the number was derived when it was computed.",
        "The answer is a bare value with no hedging between alternatives.",
    ]


class Graded(BaseModel):
    trajectory: Trajectory
    correct: bool
    judge_score: float


# --- worlds -------------------------------------------------------------------------------------------


# {{docs-fragment build_world}}
@world_env.task(cache="auto")
async def build_world(world_id: str) -> flyte.io.Dir:
    root = Path(tempfile.mkdtemp(prefix=f"world-{world_id}-"))
    build_world_files(world_id, root)
    return await flyte.io.Dir.from_local(str(root))
# {{/docs-fragment build_world}}


# --- one trial: generate ---------------------------------------------------------------------------


# {{docs-fragment run_trial}}
@trial_env.task(cache="auto", retries=API_RETRIES, timeout=flyte.Timeout(max_runtime=timedelta(minutes=6)))
async def run_trial(task: TaskSpec, policy: Policy, harness: Harness, world: flyte.io.Dir, pass_idx: int) -> Trajectory:
    """One attempt. The harness's own deadline ends a stuck trial as a *recorded* agent_timeout; the task
    timeout above is only the backstop for anything that ignores asyncio (a blocking C call, a wedged pod)."""
    import anthropic

    trial_id = f"{task.task_id}-{harness.version}-{policy.model}-p{pass_idx}"
    root = Path(await world.download(tempfile.mkdtemp(prefix="trial-")))  # private copy: writes are isolated
    traj = Trajectory(
        trial_id=trial_id, task_id=task.task_id, world_id=task.world_id, task_type=task.task_type,
        harness_version=harness.version, model=policy.model, pass_idx=pass_idx, stop_reason="error",
        produced_by_run=flyte.ctx().action.run_name or "",
    )
    t0 = asyncio.get_running_loop().time()
    try:
        await asyncio.wait_for(run_agent(task, policy, harness, World(root, harness, trial_id), traj),
                               timeout=harness.agent_timeout_s)
    except TimeoutError:
        traj.stop_reason = "agent_timeout"  # Mercor: "Anything without a timeout will eventually hang a rollout"
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
        raise  # transient: Flyte retries the whole trial with backoff
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise flyte.errors.NonRecoverableError(f"bad API credentials: {e}")  # fail fast, don't burn retries
    except anthropic.BadRequestError as e:
        traj.stop_reason = "error"  # deterministic: retrying won't help, so record it
        traj.justification = str(e)[:500]
    traj.elapsed_s = round(asyncio.get_running_loop().time() - t0, 1)
    return traj
# {{/docs-fragment run_trial}}


# --- grade -------------------------------------------------------------------------------------------


# {{docs-fragment grade}}
@flyte.trace
async def judge(traj: Trajectory, task: TaskSpec, rubric: Rubric) -> float:
    """LLM judge. Traced, so a retried grading task replays the score it already paid for."""
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=4)
    schema = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"],
              "additionalProperties": False}
    resp = await client.messages.create(
        model="claude-haiku-5-5", max_tokens=1500,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": (
            "Score this analyst answer from 0 to 1: the fraction of rubric criteria it meets.\n"
            f"Rubric: {rubric.criteria}\nQuestion: {task.prompt}\n"
            f"Answer: {traj.answer}\nJustification: {traj.justification}")}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), '{"score": 0}')
    return max(0.0, min(1.0, float(json.loads(text)["score"])))


@grade_env.task(cache="auto", retries=API_RETRIES)
async def grade(traj: Trajectory, task: TaskSpec, rubric: Rubric) -> Graded:
    correct = verify(task, traj.answer)  # the reward: deterministic
    score = await judge(traj, task, rubric) if traj.stop_reason == "submitted" else 0.0
    return Graded(trajectory=traj, correct=correct, judge_score=score)
# {{/docs-fragment grade}}


# --- summarize ------------------------------------------------------------------------------------


PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


class EvalSummary(BaseModel):
    model: str
    harness_version: str
    rubric_version: str
    n_trials: int
    pass_at_1: float
    pass_at_1_std_over_passes: float
    by_task_type: dict[str, float]
    stop_reasons: dict[str, int]
    tool_calls: int
    tool_error_rate: float
    tool_errors_by_tool: dict[str, int]
    masked_fraction: float  # trials a Mercor-style masking rule would drop from the gradient
    mean_turns: float
    judge_mean: float
    est_cost_usd: float
    cells_executed_this_run: int
    cells_reused: int


def _masked(graded: list[Graded]) -> int:
    """Mercor's rule (tito_harbor_generator.py): any error masks the prompt group; >= 2 timeouts mask it;
    fewer timeouts mask only themselves. Here a group is one task's passes."""
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for g in graded:
        groups[g.trajectory.task_id].append(g.trajectory)
    masked = 0
    for trajs in groups.values():
        errors = sum(t.stop_reason == "error" for t in trajs)
        timeouts = sum(t.stop_reason == "agent_timeout" for t in trajs)
        masked += len(trajs) if errors or timeouts >= 2 else timeouts
    return masked


def summarize(graded: list[Graded], policy: Policy, harness: Harness, rubric: Rubric, run_name: str) -> EvalSummary:
    trajs = [g.trajectory for g in graded]
    by_pass: dict[int, list[bool]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for g in graded:
        by_pass[g.trajectory.pass_idx].append(g.correct)
        by_type[g.trajectory.task_type].append(g.correct)
    pass_means = [statistics.fmean(v) for v in by_pass.values()]
    steps = [s for t in trajs for s in t.steps]
    errs = Counter(s.tool for s in steps if s.error)
    p_in, p_out = PRICE_PER_MTOK.get(policy.model, (0.0, 0.0))
    cost = sum(((t.input_tokens - t.cached_input_tokens) + 0.1 * t.cached_input_tokens) * p_in + t.output_tokens * p_out
               for t in trajs) / 1e6
    executed = sum(t.produced_by_run == run_name for t in trajs)
    return EvalSummary(
        model=policy.model, harness_version=harness.version, rubric_version=rubric.version, n_trials=len(graded),
        pass_at_1=round(statistics.fmean(g.correct for g in graded), 4) if graded else 0.0,
        pass_at_1_std_over_passes=round(statistics.pstdev(pass_means), 4) if len(pass_means) > 1 else 0.0,
        by_task_type={k: round(statistics.fmean(v), 3) for k, v in sorted(by_type.items())},
        stop_reasons=dict(Counter(t.stop_reason for t in trajs)),
        tool_calls=len(steps),
        tool_error_rate=round(sum(errs.values()) / max(len(steps), 1), 3),
        tool_errors_by_tool=dict(errs.most_common()),
        masked_fraction=round(_masked(graded) / max(len(graded), 1), 3),
        mean_turns=round(statistics.fmean(t.turns for t in trajs), 2) if trajs else 0.0,
        judge_mean=round(statistics.fmean(g.judge_score for g in graded), 3) if graded else 0.0,
        est_cost_usd=round(cost, 4),
        cells_executed_this_run=executed,
        cells_reused=len(trajs) - executed,
    )


# --- evaluate one (policy, harness) -------------------------------------------------------------------


# {{docs-fragment evaluate}}
@driver_env.task
async def evaluate(
    policy: Policy, harness: Harness, n_worlds: int = 6, passes: int = 3, rubric: Rubric = Rubric()
) -> tuple[EvalSummary, list[Graded]]:
    world_ids = [f"w{i}" for i in range(n_worlds)]
    worlds = dict(zip(world_ids, await asyncio.gather(*(build_world(w) for w in world_ids))))
    tasks = [t for w in world_ids for t in make_tasks(w)]

    async def cell(task: TaskSpec, p: int) -> Graded | None:
        try:
            traj = await run_trial(task, policy, harness, worlds[task.world_id], p)
        except Exception as e:  # retries exhausted or the backstop timeout: a gap in the grid, not a crash
            traj = Trajectory(trial_id=f"{task.task_id}-p{p}", task_id=task.task_id, world_id=task.world_id,
                              task_type=task.task_type, harness_version=harness.version, model=policy.model,
                              pass_idx=p, stop_reason="error", justification=str(e)[:300])
        return await grade(traj, task, rubric)

    with flyte.group(f"{policy.model}-{harness.version}"):
        graded = [g for g in await asyncio.gather(*(cell(t, p) for t in tasks for p in range(passes))) if g]
    return summarize(graded, policy, harness, rubric, flyte.ctx().action.run_name or ""), graded
# {{/docs-fragment evaluate}}


# --- audit: let a model read the failures --------------------------------------------------------------


class Finding(BaseModel):
    tool: str
    symptom: str
    evidence: str
    suggested_fix: str


class Audit(BaseModel):
    findings: list[Finding]


@flyte.trace
async def audit_failures(graded: list[Graded], max_trajectories: int = 40) -> Audit:
    """Mercor ran a coding agent over every trace before training. This is the same move, as one call."""
    import anthropic

    failed = [g.trajectory for g in graded if not g.correct][:max_trajectories]
    digest = []
    for t in failed:
        def clip(x: str, n: int) -> str:  # say so when *we* cut the text, or the auditor blames the tool
            return x if len(x) <= n else x[:n] + " …[clipped for this digest]"

        steps = "; ".join(f"{s.tool}({clip(s.input, 80)}) -> {clip(s.error or s.output, 220)!r}" for s in t.steps[:10])
        digest.append(f"[{t.trial_id} | {t.task_type} | stop={t.stop_reason} | answer={t.answer}] {steps}")
    schema = {
        "type": "object",
        "properties": {"findings": {"type": "array", "items": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in ("tool", "symptom", "evidence", "suggested_fix")},
            "required": ["tool", "symptom", "evidence", "suggested_fix"], "additionalProperties": False}}},
        "required": ["findings"], "additionalProperties": False,
    }
    client = anthropic.AsyncAnthropic(max_retries=4)
    resp = await client.messages.create(
        model="claude-sonnet-5-5", max_tokens=8000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        system="You audit an agent harness. From failed trajectories, list suspected bugs in the *tools or "
               "environment* (not the model's reasoning), one finding per distinct bug, citing trial ids.",
        messages=[{"role": "user", "content": "\n".join(digest)}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), '{"findings": []}')
    return Audit(**json.loads(text))


# --- reports + entrypoints ------------------------------------------------------------------------------


def _table(summaries: list[EvalSummary]) -> str:
    head = ("<tr><th>model</th><th>harness</th><th>pass@1</th><th>± over passes</th><th>by task type</th>"
            "<th>stop reasons</th><th>tool error rate</th><th>errors by tool</th><th>masked</th><th>turns</th>"
            "<th>judge</th><th>est. $</th><th>cells run / reused</th></tr>")
    rows = "".join(
        f"<tr><td>{s.model}</td><td>{s.harness_version}</td><td>{s.pass_at_1:.1%}</td>"
        f"<td>{s.pass_at_1_std_over_passes:.1%}</td><td>{s.by_task_type}</td><td>{s.stop_reasons}</td>"
        f"<td>{s.tool_error_rate:.1%}</td><td>{s.tool_errors_by_tool}</td><td>{s.masked_fraction:.0%}</td>"
        f"<td>{s.mean_turns}</td><td>{s.judge_mean}</td><td>{s.est_cost_usd}</td>"
        f"<td>{s.cells_executed_this_run} / {s.cells_reused}</td></tr>"
        for s in summaries
    )
    return f"<table border=1 cellpadding=4>{head}{rows}</table>"


async def warm_worlds(n_worlds: int) -> None:
    """Build every world once before fanning out. Two evaluations that call the same cached task at the same
    moment can both miss the cache and both run; whichever output loses then keys a different set of cells."""
    await asyncio.gather(*(build_world(f"w{i}") for i in range(n_worlds)))


# {{docs-fragment compare}}
@driver_env.task(report=True)
async def compare_harnesses(
    policy: Policy = Policy(), harnesses: list[Harness] = [HARNESS_V1, HARNESS_V2], n_worlds: int = 6,
    passes: int = 3, rubric: Rubric = Rubric(),
) -> list[EvalSummary]:
    await warm_worlds(n_worlds)
    results = await asyncio.gather(*(evaluate(policy, h, n_worlds, passes, rubric) for h in harnesses))
    summaries = [r[0] for r in results]
    audit = await audit_failures(results[0][1])  # what would a reader of v1's traces flag?
    findings = "".join(f"<li><b>{f.tool}</b>: {f.symptom} <i>({f.evidence})</i> → {f.suggested_fix}</li>"
                       for f in audit.findings)
    await flyte.report.replace.aio(
        f"<h2>Harness comparison · {policy.model}</h2>{_table(summaries)}"
        f"<h3>Audit of {harnesses[0].version} failures</h3><ul>{findings}</ul>", do_flush=True)
    return summaries
# {{/docs-fragment compare}}


# {{docs-fragment leaderboard}}
@driver_env.task(report=True)
async def leaderboard(
    policies: list[Policy] = [Policy(model="claude-haiku-5-5"), Policy(model="claude-sonnet-5-5")],
    harness: Harness = HARNESS_V2, n_worlds: int = 6, passes: int = 3, rubric: Rubric = Rubric(),
) -> list[EvalSummary]:
    """Add a model to the list and only its cells run; every other model's cells come back from cache."""
    await warm_worlds(n_worlds)
    results = await asyncio.gather(*(evaluate(p, harness, n_worlds, passes, rubric) for p in policies))
    summaries = sorted((r[0] for r in results), key=lambda s: -s.pass_at_1)
    await flyte.report.replace.aio(f"<h2>Leaderboard · harness {harness.version}</h2>{_table(summaries)}",
                                   do_flush=True)
    return summaries
# {{/docs-fragment leaderboard}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(compare_harnesses).url)
