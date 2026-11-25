# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes", "msgpack"]
# ///
"""Grade an existing set of rollouts in shards, one GPU per (judge, shard), anywhere you have GPUs.

Rollouts and grades are separate steps on purpose: a rollout file from one cluster can be graded on another, by a
different judge, without running a single episode again. Each shard serves its judge with vLLM on one GPU and runs
LAB's grader over its slice. Results come back inline and compact (one pass/fail string per rollout), so they can
be read from a laptop that can't reach the cluster's object store.

    python grade_shards.py --rollouts sweep_rollouts.jsonl.gz --shards 3
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
from pathlib import Path

import flyte
import flyte.io
from pydantic import BaseModel

import lab_harness as lh
import servers
from config import PROPAGATE, cpu_env_kwargs, gpu_env_kwargs
from images import LAB_COMMIT

shard_env = flyte.TaskEnvironment(name="legal-rl-grade-shard", env_vars=PROPAGATE, **gpu_env_kwargs())
driver_env = flyte.TaskEnvironment(name="legal-rl-grade-driver", env_vars=PROPAGATE, depends_on=[shard_env],
                                   **cpu_env_kwargs(cpu=1, memory="2Gi"))


class ShardJudge(BaseModel):
    judge: lh.JudgeSpec
    samples: int = 1
    sample_temperature: float = 1.0
    max_model_len: int = 65536


class ShardResult(BaseModel):
    judge: str
    shard: int
    verdicts: dict[str, list[str]]  # rollout_id -> one "PFE..." string per judge sample, criteria in id order
    criteria: dict[str, list[str]]  # task_id -> criterion ids in that order
    groups: dict[str, dict[str, list[str]]] = {}  # task_id -> issue groups (shard 0 of the first judge only)


@shard_env.task(cache="auto", retries=3)
async def grade_shard(rollouts: flyte.io.File, sj: ShardJudge, shard: int, n_shards: int,
                      make_groups: bool = False) -> ShardResult:
    with gzip.open(await rollouts.download(), "rt") as f:
        rs = [lh.Rollout.model_validate_json(line) for i, line in enumerate(f) if i % n_shards == shard]
    tasks = sorted({r.task_id for r in rs})
    lab = lh.fetch_lab(tasks, LAB_COMMIT, Path("/tmp/lab"))
    srv = servers.ServerSpec(served_name=sj.judge.name, hf_id=sj.judge.hf_id, gpus=[0],
                             max_model_len=sj.max_model_len, gpu_memory_utilization=0.9)
    await servers.ensure(srv)
    sem = asyncio.Semaphore(24)
    out: dict[str, list[str]] = {}

    async def one(r: lh.Rollout):
        task, _ = lh.load_task(lab, r.task_id)
        strs = []
        for k in range(sj.samples):
            spec = sj.judge.model_copy(update={"temperature": sj.judge.temperature if k == 0 else sj.sample_temperature})
            async with sem:
                v = await asyncio.to_thread(lh.grade, r, task, lh.OpenAICompatJudge(spec, srv.url), spec.name, 12, k)
            strs.append("".join({"pass": "P", "fail": "F"}.get(v.verdicts.get(c["id"]), "E") for c in task.criteria))
        out[r.rollout_id] = strs

    await asyncio.gather(*[one(r) for r in rs])
    criteria = {t: [c["id"] for c in lh.load_task(lab, t)[0].criteria] for t in tasks}
    groups = {}
    if make_groups:
        import openai

        client = openai.OpenAI(base_url=srv.url, api_key="x", timeout=900)
        for t in tasks:
            groups[t] = await asyncio.to_thread(lh.group_issues, lh.load_task(lab, t)[0], client, sj.judge.name)
    return ShardResult(judge=sj.judge.name, shard=shard, verdicts=out, criteria=criteria, groups=groups)


@driver_env.task
async def grade_all(rollouts: flyte.io.File, judges: list[ShardJudge], n_shards: int) -> list[ShardResult]:
    return list(await asyncio.gather(*[grade_shard(rollouts, sj, s, n_shards, make_groups=(i == 0))
                                       for i, sj in enumerate(judges) for s in range(n_shards)]))


DEMO_JUDGES = [
    ShardJudge(judge=lh.JudgeSpec(name="gpt-oss-20b", hf_id="openai/gpt-oss-20b", reasoning_effort="low"), samples=2),
    # Nemotron 3 Nano 30B's FP8 checkpoint wouldn't start on the demo cluster's L40S, and its BF16 one doesn't fit a
    # 48 GB GPU, so the second family's judge is its 4B sibling.
    ShardJudge(judge=lh.JudgeSpec(name="nemotron-3-nano-4b", hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                                  thinking=False), samples=1),
]


def to_sweep_jsonl(rollouts_path: str, results: list[ShardResult], out_path: str) -> None:
    """Merge shard results into the sweep format reward_audit.py reads."""
    criteria = {t: ids for r in results for t, ids in r.criteria.items()}
    # Issue groups come from the first judge's shards only, so adding a judge later doesn't change what `h` measures.
    first = results[0].judge if results else None
    groups = {t: g for r in results if r.judge == first for t, g in r.groups.items()}
    grades: dict[str, list[dict]] = {}
    for res in results:
        for rid, strs in res.verdicts.items():
            for k, s in enumerate(strs):
                grades.setdefault(rid, []).append({"judge": res.judge, "judge_sample": k, "rollout_id": rid,
                                                   "verdicts": {}, "_s": s})
    with gzip.open(rollouts_path, "rt") as f, gzip.open(out_path, "wt") as g:
        for t, gr in groups.items():
            g.write(json.dumps({"kind": "groups", "task_id": t, "groups": gr}) + "\n")
        for line in f:
            r = json.loads(line)
            ids = criteria.get(r["task_id"], [])
            gs = []
            for x in grades.get(r["rollout_id"], []):
                s = x.pop("_s")
                x["task_id"] = r["task_id"]
                x["verdicts"] = {c: {"P": "pass", "F": "fail"}.get(ch, "error") for c, ch in zip(ids, s)}
                gs.append(x)
            r.pop("outputs_b64", None)
            g.write(json.dumps({"kind": "rollout", **r, "grades": gs}) + "\n")


def fetch_results(run_names: list[str]) -> list[ShardResult]:
    """Read grade_all's outputs back from finished runs (a list of pydantic models, msgpack-encoded)."""
    import msgpack
    import flyte.remote

    def dec(lit):
        if lit.HasField("collection"):
            return [dec(x) for x in lit.collection.literals]
        return msgpack.unpackb(lit.scalar.binary.value)

    out: list[ShardResult] = []
    for name in run_names:
        d = flyte.remote.ActionDetails.get(run_name=name, name="a0")
        out += [ShardResult(**x) for x in dec(d.output_literals()["o0"])]
    return out


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="grade: launch grade_all. merge: fetch finished runs, write the sweep "
                                             "file reward_audit.py reads, and print the audit.")
    ap.add_argument("what", nargs="?", choices=["grade", "merge"], default="grade")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--judges", default="", help="comma-separated judge names from DEMO_JUDGES (default: all)")
    ap.add_argument("--max-model-len", type=int, default=0, help="e.g. 32768 on a 24 GB GPU")
    ap.add_argument("--runs", default="", help="merge: comma-separated grade_all run names; the first one's issue groups are used")
    ap.add_argument("--out", default="sweep_graded.jsonl.gz", help="merge: the merged sweep file")
    ap.add_argument("--judge-a", default="gpt-oss-20b")
    ap.add_argument("--judge-b", default="nemotron-3-nano-4b")
    args = ap.parse_args()
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    if args.what == "merge":
        import reward_audit

        to_sweep_jsonl(args.rollouts, fetch_results(args.runs.split(",")), args.out)
        print(_json.dumps(reward_audit.report(args.out, args.judge_a, args.judge_b or None), indent=1))
    else:
        judges = [j for j in DEMO_JUDGES if not args.judges or j.judge.name in args.judges.split(",")]
        if args.max_model_len:
            judges = [j.model_copy(update={"max_model_len": args.max_model_len}) for j in judges]
        run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(
            grade_all, flyte.io.File.from_local_sync(args.rollouts), judges, args.shards)
        print(run.name, run.url)
