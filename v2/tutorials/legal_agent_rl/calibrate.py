# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic"]
# ///
"""Grade a calibration subset with LAB's standard judges, so open-judge numbers have a known relationship to LAB's.

LAB's standard profile grades every criterion with ``claude-sonnet-4-6`` and ``gpt-5.5`` and averages them. This runs
``lab_core``'s own ``Judge`` (unchanged) on deliverables exported from a sweep, one cached action per rollout and
judge, on a cluster that holds the API keys.

    python calibrate.py --export export.json.gz      # the file audit.py's report carries
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import tempfile
from pathlib import Path

import flyte
import flyte.io

import lab_harness as lh
from config import cpu_env_kwargs
from images import LAB_COMMIT

ANTHROPIC = os.getenv("ANTHROPIC_SECRET", "ANTHROPIC_API_KEY")
OPENAI = os.getenv("OPENAI_SECRET", "OPENAI_API_KEY")
LAB_JUDGES = ["claude-sonnet-4-6", "gpt-5.5"]

calib_env = flyte.TaskEnvironment(
    name="legal-rl-calibrate",
    secrets=[flyte.Secret(key=ANTHROPIC, as_env_var="ANTHROPIC_API_KEY"), flyte.Secret(key=OPENAI, as_env_var="OPENAI_API_KEY")],
    **{**cpu_env_kwargs(cpu=4, memory="8Gi"), "queue": None})


@calib_env.task(cache="auto", retries=3)
async def grade_lab(rollouts: list[lh.Rollout], judge: str) -> list[lh.Verdicts]:
    from lab_core.evaluation.judge import Judge

    lab = lh.fetch_lab(sorted({r.task_id for r in rollouts}), LAB_COMMIT, Path("/tmp/lab"))
    j = Judge(judge)
    sem = asyncio.Semaphore(8)

    async def one(r):
        async with sem:
            task, _ = lh.load_task(lab, r.task_id)
            return await asyncio.to_thread(lh.grade, r, task, j, judge, 8)

    return list(await asyncio.gather(*[one(r) for r in rollouts]))


@calib_env.task
async def calibrate(rollouts: list[lh.Rollout]) -> str:
    chunks = [rollouts[i:i + 8] for i in range(0, len(rollouts), 8)]
    results = await asyncio.gather(*[grade_lab(c, j) for j in LAB_JUDGES for c in chunks])
    return json.dumps([v.model_dump(exclude={"reasoning"}) for vs in results for v in vs])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    args = ap.parse_args()
    exp = json.load(gzip.open(args.export, "rt"))
    rs = [lh.Rollout(rollout_id=c["rollout_id"], task_id=c["task_id"], policy=c["policy"], harness=c["harness"],
                     sample=c["sample"], finish_reason=c["finish_reason"], turns=c["turns"], input_tokens=0,
                     output_tokens=0, wall_s=0, outputs_b64=c["outputs_b64"]) for c in exp["calibration"]]
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    run = flyte.run(calibrate, rs)
    print(run.name, run.url)
