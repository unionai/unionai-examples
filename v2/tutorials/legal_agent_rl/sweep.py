# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Base-model rollouts on LAB tasks, graded by open-weight judges, on one GPU node.

This is the data behind every post in the series: three US open-weight policies, G samples per task, LAB's own
harness and grader, and two judges from different families so the judge can be measured too. Two cached tasks, each
on a whole node:

    rollouts   gpt-oss-120b (GPUs 0-3), gpt-oss-20b (4-5), Nemotron 3 Nano 30B-A3B (6-7), one server each
    grades     judge A gpt-oss-120b (GPUs 0-3, three gradings per rollout), judge B Nemotron (4-7)

The node comes from a GPU market and can be taken back at any time, so each rollout and each grade is a
``@flyte.trace``: when a task retries on a new node, finished work replays from the record and only the rest runs.
Both tasks are cached on their inputs, and grading is its own task, so changing a judge never re-runs a rollout.

    CLUSTER=amd GPU_QUEUE=<queue> python sweep.py --tasks 48 --samples 8
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.report
from pydantic import BaseModel, Field

import lab_harness as lh
import servers
from config import CLUSTER, PROPAGATE, cpu_env_kwargs, gpu_env_kwargs
from images import LAB_COMMIT

node_env = flyte.TaskEnvironment(name="legal-rl-node", env_vars=PROPAGATE, **gpu_env_kwargs())
driver_env = flyte.TaskEnvironment(name="legal-rl-driver", env_vars=PROPAGATE, depends_on=[node_env],
                                   **cpu_env_kwargs())


# --- what to run -----------------------------------------------------------------------------------------------


class PolicyRun(BaseModel):
    policy: lh.Policy
    server: servers.ServerSpec


class JudgeRun(BaseModel):
    judge: lh.JudgeSpec
    server: servers.ServerSpec | None = None  # None for lab_core's hosted judges
    samples: int = 1  # >1 re-grades every rollout, to measure the judge's own noise
    sample_temperature: float = 1.0  # used for samples after the first, which run at the spec's temperature


class SweepConfig(BaseModel):
    task_ids: list[str]
    samples: int = 8  # G: rollouts per (task, policy), as in Harvey's recipe
    harness: lh.Harness = lh.Harness()
    policies: list[PolicyRun]
    judges: list[JudgeRun]
    rollout_concurrency: int = 48  # per policy server
    grade_concurrency: int = 48  # rollouts being graded at once, per judge
    criterion_parallel: int = 16  # judge calls per rollout at once (LAB's --parallel)
    group_issues_with: str = ""  # judge name whose model proposes issue groups; "" = the first judge


class SweepResult(BaseModel):
    data: flyte.io.File  # sweep.jsonl.gz: one line per rollout, verdict set and issue grouping
    n_rollouts: int
    n_verdicts: int
    summary: dict = Field(default_factory=dict)


# {{docs-fragment traced}}
@flyte.trace
async def traced_rollout(lab_root: str, task_id: str, policy: lh.Policy, harness: lh.Harness, base_url: str,
                         sample: int) -> lh.Rollout:
    return await asyncio.to_thread(lh.run_rollout, Path(lab_root), task_id, policy, harness, base_url, sample)


@flyte.trace
async def traced_grade(lab_root: str, rollout: lh.Rollout, judge: lh.JudgeSpec, base_url: str, judge_sample: int,
                       temperature: float, parallel: int) -> lh.Verdicts:
    task, _ = lh.load_task(Path(lab_root), rollout.task_id)
    spec = judge.model_copy(update={"temperature": temperature})
    j = lh.OpenAICompatJudge(spec, base_url) if judge.kind == "openai_compat" else _lab_judge(judge.name)
    v = await asyncio.to_thread(lh.grade, rollout, task, j, judge.name, parallel, judge_sample)
    return v
# {{/docs-fragment traced}}


@flyte.trace
async def traced_groups(lab_root: str, task_id: str, model: str, base_url: str) -> dict[str, list[str]]:
    import openai

    task, _ = lh.load_task(Path(lab_root), task_id)
    client = openai.OpenAI(base_url=base_url, api_key="local", timeout=900)
    return await asyncio.to_thread(lh.group_issues, task, client, model)


def _lab_judge(name: str):
    from lab_core.evaluation.judge import Judge

    return Judge(name)


def _strip(r: lh.Rollout) -> dict:
    return r.model_dump(exclude={"tokens"})


class RolloutSet(BaseModel):
    data: flyte.io.File  # rollouts.jsonl.gz: one Rollout per line, deliverables included
    n: int
    summary: dict = Field(default_factory=dict)


async def _report_loop(label: str, progress: dict, total: int, t0: float):
    while progress["done"] < total:
        await flyte.report.replace.aio(
            f"<h3>{label}</h3><p>{progress['done']}/{total} done, {(time.time() - t0) / 60:.0f} min</p>"
            f"<pre>{json.dumps(servers.gpu_report(), indent=1)}</pre>", do_flush=True)
        await asyncio.sleep(60)


# {{docs-fragment sweep}}
@node_env.task(cache="auto", retries=6, report=True)
async def rollouts(cfg: SweepConfig) -> RolloutSet:
    """Every (task, policy, sample) episode, on one node, durable across the node being taken back."""
    t0 = time.time()
    lab = str(lh.fetch_lab(cfg.task_ids, LAB_COMMIT, Path("/tmp/lab")))
    loads = dict(zip([p.server.served_name for p in cfg.policies],
                     await asyncio.gather(*[servers.ensure(p.server) for p in cfg.policies])))
    out: list[lh.Rollout] = []
    progress = {"done": 0}
    total = len(cfg.task_ids) * cfg.samples * len(cfg.policies)

    async def run_policy(p: PolicyRun):
        sem = asyncio.Semaphore(cfg.rollout_concurrency)

        async def one(t: str, s: int):
            async with sem:
                out.append(await traced_rollout(lab, t, p.policy, cfg.harness, p.server.url, s))
            progress["done"] += 1

        await asyncio.gather(*[one(t, s) for s in range(cfg.samples) for t in cfg.task_ids])

    reporter = asyncio.create_task(_report_loop("rollouts", progress, total, t0))
    await asyncio.gather(*[run_policy(p) for p in cfg.policies])
    reporter.cancel()
    path = Path("/tmp/rollouts.jsonl.gz")
    with gzip.open(path, "wt") as f:
        for r in out:
            f.write(json.dumps(_strip(r)) + "\n")
    summary = {"minutes": round((time.time() - t0) / 60, 1), "server_load_s": loads,
               "finish_reasons": _count(r.finish_reason for r in out)}
    return RolloutSet(data=await flyte.io.File.from_local(str(path)), n=len(out), summary=summary)


@node_env.task(cache="auto", retries=6, report=True)
async def grades(rs: RolloutSet, cfg: SweepConfig) -> SweepResult:
    """Every rollout graded by every judge (and re-graded, for the judge's own noise). A separate cached task, so
    changing a judge re-runs grading and never a rollout."""
    t0 = time.time()
    lab = str(lh.fetch_lab(cfg.task_ids, LAB_COMMIT, Path("/tmp/lab")))
    with gzip.open(await rs.data.download(), "rt") as f:
        rollouts_ = [lh.Rollout.model_validate_json(line) for line in f]
    judges = cfg.judges
    await asyncio.gather(*[servers.ensure(j.server) for j in judges if j.server])
    gj = next((j for j in judges if j.judge.name == cfg.group_issues_with), judges[0])
    groups = dict(zip(cfg.task_ids, await asyncio.gather(*[
        traced_groups(lab, t, gj.judge.name, gj.server.url) for t in cfg.task_ids])))
    verdicts: list[lh.Verdicts] = []
    progress = {"done": 0}
    sems = {j.judge.name: asyncio.Semaphore(cfg.grade_concurrency) for j in judges}

    async def one(r: lh.Rollout, j: JudgeRun, k: int):
        async with sems[j.judge.name]:
            temp = j.judge.temperature if k == 0 else j.sample_temperature
            verdicts.append(await traced_grade(lab, r, j.judge, j.server.url if j.server else "", k, temp,
                                               cfg.criterion_parallel))
        progress["done"] += 1

    work = [(r, j, k) for k in range(max(j.samples for j in judges)) for j in judges if k < j.samples for r in rollouts_]
    reporter = asyncio.create_task(_report_loop("grades", progress, len(work), t0))
    await asyncio.gather(*[one(r, j, k) for r, j, k in work])  # every rollout's first grading before any re-grade
    reporter.cancel()
    path = Path("/tmp/sweep.jsonl.gz")
    by_rollout: dict[str, list[dict]] = {}
    for v in verdicts:
        by_rollout.setdefault(v.rollout_id, []).append(v.model_dump())
    with gzip.open(path, "wt") as f:
        for t, g in groups.items():
            f.write(json.dumps({"kind": "groups", "task_id": t, "groups": g}) + "\n")
        for r in rollouts_:
            f.write(json.dumps({"kind": "rollout", **_strip(r), "grades": by_rollout.get(r.rollout_id, [])}) + "\n")
    summary = {**rs.summary, "grade_minutes": round((time.time() - t0) / 60, 1), "cluster": CLUSTER}
    return SweepResult(data=await flyte.io.File.from_local(str(path)), n_rollouts=len(rollouts_),
                       n_verdicts=len(verdicts), summary=summary)


@driver_env.task(cache="auto")
async def sweep(cfg: SweepConfig) -> SweepResult:
    return await grades(await rollouts(cfg), cfg)
# {{/docs-fragment sweep}}


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --- the default sweep -----------------------------------------------------------------------------------------

# Harvey trained on LAB's 24 practice areas, leaving out contracts, firm-knowledge and diligence (2,010 - 498 - 250 -
# 11 = 1,251 tasks). We sample from the same pool: the tasks with at most 40 criteria and one deliverable, two per
# practice area, with a fixed seed. Our split, not Harvey's hold-out set (LAB doesn't publish one).
EXCLUDED_AREAS = {"contracts", "firm-knowledge", "diligence"}


def pick_tasks(catalog: list[dict], n: int, seed: int = 0, max_criteria: int = 40) -> list[str]:
    rng = random.Random(seed)
    pool = [r for r in catalog if r["area"] not in EXCLUDED_AREAS and r["n"] <= max_criteria and r["ndeliv"] == 1]
    by_area: dict[str, list[str]] = {}
    for r in sorted(pool, key=lambda r: r["id"]):
        by_area.setdefault(r["area"], []).append(r["id"])
    picked: list[str] = []
    while len(picked) < n and any(by_area.values()):
        for area in sorted(by_area):
            if by_area[area] and len(picked) < n:
                picked.append(by_area[area].pop(rng.randrange(len(by_area[area]))))
    return sorted(picked)


def default_config(task_ids: list[str], samples: int) -> SweepConfig:
    def srv(name, hf, gpus, **kw):
        return servers.ServerSpec(served_name=name, hf_id=hf, gpus=gpus, **kw)

    # Rollout node: one server per policy. Grading node: the whole node for the two judges.
    gpt120 = srv("gpt-oss-120b", "openai/gpt-oss-120b", [0, 1, 2, 3], max_model_len=131072)
    gpt20 = srv("gpt-oss-20b", "openai/gpt-oss-20b", [4, 5], max_model_len=131072)
    nemo = srv("nemotron-3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", [6, 7], max_model_len=131072)
    judge_a = srv("gpt-oss-120b", "openai/gpt-oss-120b", [0, 1, 2, 3], max_model_len=131072)
    judge_b = srv("nemotron-3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", [4, 5, 6, 7],
                  max_model_len=131072)
    policies = [
        PolicyRun(policy=lh.Policy(name="gpt-oss-20b", hf_id=gpt20.hf_id, reasoning_effort="medium"), server=gpt20),
        PolicyRun(policy=lh.Policy(name="nemotron-3-nano-30b", hf_id=nemo.hf_id), server=nemo),
        PolicyRun(policy=lh.Policy(name="gpt-oss-120b", hf_id=gpt120.hf_id, reasoning_effort="medium"), server=gpt120),
    ]
    judges = [
        JudgeRun(judge=lh.JudgeSpec(name="gpt-oss-120b", hf_id=judge_a.hf_id, reasoning_effort="low"), server=judge_a,
                 samples=3),
        # Thinking off: a verdict needs a few hundred tokens of reasoning, not a few thousand.
        JudgeRun(judge=lh.JudgeSpec(name="nemotron-3-nano-30b", hf_id=judge_b.hf_id, thinking=False), server=judge_b,
                 samples=1),
    ]
    return SweepConfig(task_ids=task_ids, samples=samples, policies=policies, judges=judges)


def demo_config(task_ids: list[str], samples: int) -> SweepConfig:
    """The same sweep on one NVIDIA GPU: gpt-oss-20b is the policy and grades itself (disclosed as self-judging),
    which is enough to check that the code runs unchanged on CUDA and to see the run in the console."""
    gpt20 = servers.ServerSpec(served_name="gpt-oss-20b", hf_id="openai/gpt-oss-20b", gpus=[0],
                               max_model_len=65536, gpu_memory_utilization=0.9)
    return SweepConfig(task_ids=task_ids, samples=samples, harness=lh.Harness(max_turns=30),
                       policies=[PolicyRun(policy=lh.Policy(name="gpt-oss-20b", hf_id=gpt20.hf_id,
                                                            reasoning_effort="medium"), server=gpt20)],
                       judges=[JudgeRun(judge=lh.JudgeSpec(name="gpt-oss-20b", hf_id=gpt20.hf_id,
                                                           reasoning_effort="low"), server=gpt20, samples=2)],
                       rollout_concurrency=16, grade_concurrency=8)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=48)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--catalog", default="lab_tasks.json", help="python catalog.py writes it")
    args = ap.parse_args()
    task_ids = pick_tasks(json.load(open(args.catalog))["tasks"], args.tasks)
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    cfg = (default_config if CLUSTER == "amd" else demo_config)(task_ids, args.samples)
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(sweep, cfg)
    print(run.name, run.url)
