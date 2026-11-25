# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Base-model rollouts on LAB tasks, graded by open-weight judges, on one GPU node.

This is the data behind every post in the series: several US open-weight policies, G samples per task, LAB's own
harness and grader, and more than one judge so the judge can be measured too. Everything runs on one node:

    GPUs 0,1,5,6   gpt-oss-120b, data-parallel x4: judge A, and the strongest policy
    GPUs 2,7       Nemotron 3 Nano 30B-A3B, data-parallel x2: judge B (another family), and a policy
    GPUs 3,4       gpt-oss-20b, data-parallel x2: the policy we train

The node comes from a GPU market and can be taken back at any time, so each rollout and each grade is a
``@flyte.trace``: when the task retries on a new node, finished rollouts and grades replay from the record and only
the rest run. The task itself is cached on its inputs, so asking for the same sweep twice costs nothing.

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
    grade_concurrency: int = 24  # rollouts being graded at once, per judge
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


# {{docs-fragment sweep}}
@node_env.task(cache="auto", retries=4, report=True)
async def sweep(cfg: SweepConfig) -> SweepResult:
    """All rollouts and all grades for one config, on one node, durable across the node being taken back."""
    t0 = time.time()
    lab = str(lh.fetch_lab(cfg.task_ids, LAB_COMMIT, Path("/tmp/lab")))

    specs = {p.server.served_name: p.server for p in cfg.policies}
    specs.update({j.server.served_name: j.server for j in cfg.judges if j.server})
    loads = dict(zip(specs, await asyncio.gather(*[servers.ensure(s) for s in specs.values()])))

    judges = cfg.judges
    groups_judge = next((j for j in judges if j.judge.name == cfg.group_issues_with), judges[0])
    groups = dict(zip(cfg.task_ids, await asyncio.gather(*[
        traced_groups(lab, t, groups_judge.judge.name, groups_judge.server.url) for t in cfg.task_ids])))

    rollouts: list[lh.Rollout] = []
    verdicts: list[lh.Verdicts] = []
    progress = {"rollouts": 0, "graded": 0}
    grade_sems = {j.judge.name: asyncio.Semaphore(cfg.grade_concurrency) for j in judges}

    async def grade_all(r: lh.Rollout):
        async def one(j: JudgeRun, k: int):
            async with grade_sems[j.judge.name]:
                temp = j.judge.temperature if k == 0 else j.sample_temperature
                url = j.server.url if j.server else ""
                verdicts.append(await traced_grade(lab, r, j.judge, url, k, temp, cfg.criterion_parallel))
        await asyncio.gather(*[one(j, k) for j in judges for k in range(j.samples)])
        progress["graded"] += 1

    async def run_policy(p: PolicyRun):
        sem = asyncio.Semaphore(cfg.rollout_concurrency)
        work = [(t, s) for s in range(cfg.samples) for t in cfg.task_ids]  # sample-major: every task gets s=0 first

        async def one(t: str, s: int):
            async with sem:
                r = await traced_rollout(lab, t, p.policy, cfg.harness, p.server.url, s)
            rollouts.append(r)
            progress["rollouts"] += 1
            await grade_all(r)

        await asyncio.gather(*[one(t, s) for t, s in work])

    async def live_report():
        total = len(cfg.task_ids) * cfg.samples * len(cfg.policies)
        while progress["graded"] < total:
            await flyte.report.replace.aio(
                f"<h3>sweep</h3><p>{progress['rollouts']}/{total} rollouts, {progress['graded']}/{total} graded, "
                f"{(time.time() - t0) / 60:.0f} min</p><pre>{json.dumps(servers.gpu_report(), indent=1)}</pre>",
                do_flush=True)
            await asyncio.sleep(60)

    reporter = asyncio.create_task(live_report())
    await asyncio.gather(*[run_policy(p) for p in cfg.policies])
    reporter.cancel()

    path = Path("/tmp/sweep.jsonl.gz")
    by_rollout: dict[str, list[dict]] = {}
    for v in verdicts:
        by_rollout.setdefault(v.rollout_id, []).append(v.model_dump())
    with gzip.open(path, "wt") as f:
        for t, g in groups.items():
            f.write(json.dumps({"kind": "groups", "task_id": t, "groups": g}) + "\n")
        for r in rollouts:
            f.write(json.dumps({"kind": "rollout", **_strip(r), "grades": by_rollout.get(r.rollout_id, [])}) + "\n")
    summary = {"minutes": round((time.time() - t0) / 60, 1), "server_load_s": loads, "cluster": CLUSTER,
               "finish_reasons": _count(r.finish_reason for r in rollouts)}
    await flyte.report.replace.aio(f"<h3>sweep done</h3><pre>{json.dumps(summary, indent=1)}</pre>", do_flush=True)
    return SweepResult(data=await flyte.io.File.from_local(str(path)), n_rollouts=len(rollouts),
                       n_verdicts=len(verdicts), summary=summary)
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

    gpt120 = srv("gpt-oss-120b", "openai/gpt-oss-120b", [0, 1, 5, 6], max_model_len=131072)
    nemo = srv("nemotron-3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", [2, 7], max_model_len=131072)
    gpt20 = srv("gpt-oss-20b", "openai/gpt-oss-20b", [3, 4], max_model_len=131072)
    policies = [
        PolicyRun(policy=lh.Policy(name="gpt-oss-20b", hf_id=gpt20.hf_id, reasoning_effort="medium"), server=gpt20),
        PolicyRun(policy=lh.Policy(name="nemotron-3-nano-30b", hf_id=nemo.hf_id), server=nemo),
        PolicyRun(policy=lh.Policy(name="gpt-oss-120b", hf_id=gpt120.hf_id, reasoning_effort="medium"), server=gpt120),
    ]
    judges = [
        JudgeRun(judge=lh.JudgeSpec(name="gpt-oss-120b", hf_id=gpt120.hf_id, reasoning_effort="low"), server=gpt120,
                 samples=3),
        JudgeRun(judge=lh.JudgeSpec(name="nemotron-3-nano-30b", hf_id=nemo.hf_id), server=nemo, samples=1),
    ]
    return SweepConfig(task_ids=task_ids, samples=samples, policies=policies, judges=judges)


def demo_config(task_ids: list[str], samples: int) -> SweepConfig:
    """The same sweep on one NVIDIA GPU: Olmo 3 7B as the policy, gpt-oss-20b as the judge, sharing the card."""
    olmo = servers.ServerSpec(served_name="olmo-3-7b", hf_id="allenai/Olmo-3-7B-Instruct", gpus=[0],
                              max_model_len=32768, gpu_memory_utilization=0.42)
    judge = servers.ServerSpec(served_name="gpt-oss-20b", hf_id="openai/gpt-oss-20b", gpus=[0], port=8200,
                               max_model_len=65536, gpu_memory_utilization=0.42)
    return SweepConfig(task_ids=task_ids, samples=samples, harness=lh.Harness(max_turns=20),
                       policies=[PolicyRun(policy=lh.Policy(name="olmo-3-7b", hf_id=olmo.hf_id), server=olmo)],
                       judges=[JudgeRun(judge=lh.JudgeSpec(name="gpt-oss-20b", hf_id=judge.hf_id,
                                                           reasoning_effort="low"), server=judge, samples=2)],
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
