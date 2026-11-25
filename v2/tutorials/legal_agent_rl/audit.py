# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Run the reward audit over a sweep, in the same cluster pool as the sweep, and publish it as a report.

The audit is CPU work over verdicts that are already paid for, so changing the reward's weights (or dropping the
truncation rule) is a new, seconds-long run of this task, not a new sweep.

The report also carries a compact export (every verdict, plus the deliverables of a calibration subset) so the
numbers can leave a cluster whose object store you can't read from your laptop.

    CLUSTER=amd GPU_QUEUE=<queue> python audit.py --sweep-run <run name>
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import random

import flyte
import flyte.io
import flyte.report

import reward_audit as ra
from config import PROPAGATE, cpu_env_kwargs
from lab_harness import RewardWeights

audit_env = flyte.TaskEnvironment(name="legal-rl-audit", env_vars=PROPAGATE, **cpu_env_kwargs(cpu=2, memory="6Gi"))


def compact(rollouts: list[dict], groups: dict, calib_per_policy: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows = [{k: r[k] for k in ("rollout_id", "task_id", "policy", "sample", "finish_reason", "turns", "output_tokens",
                               "input_tokens", "deliverable_chars", "wall_s", "tool_metrics")}
            | {"grades": [{"judge": g["judge"], "s": g["judge_sample"],
                           "v": "".join({"pass": "P", "fail": "F"}.get(g["verdicts"][c], "E") for c in sorted(g["verdicts"])),
                           "ids": sorted(g["verdicts"])} for g in r["grades"]]} for r in rollouts]
    calib = []
    by_policy: dict[str, list[dict]] = {}
    for r in rollouts:
        if r["deliverable_chars"] > 0 and r["sample"] == 0:
            by_policy.setdefault(r["policy"], []).append(r)
    for pol, rs in sorted(by_policy.items()):
        for r in rng.sample(rs, min(calib_per_policy, len(rs))):
            calib.append({k: r[k] for k in ("rollout_id", "task_id", "policy", "sample", "finish_reason", "harness",
                                            "turns", "outputs_b64")})
    return {"rollouts": rows, "groups": groups, "calibration": calib}


@audit_env.task(cache="auto", report=True)
async def audit(sweep_data: flyte.io.File, judge_a: str = "gpt-oss-120b", judge_b: str = "nemotron-3-nano-30b",
                weights: RewardWeights = RewardWeights(), calib_per_policy: int = 16) -> str:
    local = await sweep_data.download()
    out = ra.report(local, judge_a, judge_b or None, weights)
    rollouts, groups = ra.load(local)
    blob = base64.b64encode(gzip.compress(json.dumps(compact(rollouts, groups, calib_per_policy)).encode())).decode()
    await flyte.report.replace.aio(
        f"<h3>reward audit</h3><pre>{json.dumps(out, indent=1)}</pre>"
        f'<script type="application/json" id="export">{blob}</script>', do_flush=True)
    return json.dumps(out)


if __name__ == "__main__":
    import argparse

    import flyte.remote

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-run", required=True)
    ap.add_argument("--judge-a", default="gpt-oss-120b")
    ap.add_argument("--judge-b", default="nemotron-3-nano-30b", help='"" for a single-judge sweep')
    args = ap.parse_args()
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    data = flyte.remote.Run.get(name=args.sweep_run).outputs()[0].data
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(audit, data, args.judge_a, args.judge_b)
    print(run.name, run.url)
