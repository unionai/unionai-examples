# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.4",
#    "flyteplugins-union>=0.23.0",
# ]
# ///
"""SWE-Gym rollouts from your laptop, sandboxes on Union: Modal's driver, with Union under it.

This is the shape of Modal's example (helpers/job.py `run_group`): your process holds the
model client, its keys and the rollout loop; every sandbox runs remotely. Nothing runs in
the cluster on your behalf but the sandboxes themselves: no task, no app.

    Modal                                        Union (this file)
    ---------------------------------------      ---------------------------------------------
    environments = Environments()                flyte.init_from_config(cfg)
                                                 environments = UnionEnvironments(cache_testbeds=True)
    modal.Sandbox.create(image=<instance>)       a microVM on a CoW fork of the instance's testbed
    sb.exec(...)                                 sb.run(...): through Union's data proxy, as you
    ThreadPoolExecutor(rollout_workers)          a semaphore of rollout_workers
    GradingPool(grading_workers)                 a semaphore of grading_workers
    image per instance, pulled                   testbed per instance, built once and published
                                                 (SandboxVolume.from_artifact next time)

How it reaches the sandboxes: a session started outside the cluster is a run of its own,
and the plugin calls its API through the control plane's data proxy, with the token
`flyte` signed in with (the same route `flyte debug` uses to reach a pod). The data proxy
checks you may create executions in the project, and forwards to the node that hosts the
session. Sandboxes keep no network and get no secrets: model keys stay in this process.

Run, from this directory:
  python driver_laptop.py --project P --smoke
  python driver_laptop.py --project P --sampler tinker
  python driver_laptop.py --project P --sampler openai --base-url http://my-vllm:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flyte  # noqa: E402
import rollouts  # noqa: E402
import swe_gym_tasks as tasks  # noqa: E402
from union_environments import InfrastructureError, UnionEnvironments  # noqa: E402

#: Times a rollout is run again when the infrastructure, not the agent, failed.
INFRA_RETRIES = 2
#: The label every sandbox of one job carries, to find them together.
RUN_LABEL = "swe-gym-run"


class Job:
    """Modal's `Job`, the part that makes rollouts: groups of attempts at an instance, each
    rollout graded as soon as it is ready."""

    def __init__(self, envs, sampler, tokenizer, cfg, log, *, rollout_workers: int, grading_workers: int):
        self.env, self.sampler, self.tokenizer, self.cfg, self.log = envs, sampler, tokenizer, cfg, log
        self.episodes = asyncio.Semaphore(rollout_workers)
        self.graders = asyncio.Semaphore(grading_workers)

    async def _rollout(self, inst, identity):
        async with self.episodes:
            for tries in range(INFRA_RETRIES + 1):
                try:
                    return await rollouts.generate_rollout(
                        inst, self.sampler, self.tokenizer, self.env, self.cfg, self.log, identity
                    )
                except InfrastructureError as e:
                    if tries == INFRA_RETRIES:
                        raise
                    print(f"  {inst['instance_id']} attempt {identity['attempt']}: infrastructure failed, again: {e}")

    async def _grade(self, prepared, inst):
        async with self.graders:
            return await rollouts.grade_rollout(prepared, inst, self.env, self.log)

    async def run_group(self, inst, group: int) -> Dict[str, Any]:
        identity = {"phase": "rollout", "job": 0, "group": group, "policy_version": 0}
        size = self.cfg["group_size"]

        async def attempt(n: int):
            prepared = await self._rollout(inst, {**identity, "attempt": n})
            return await self._grade(prepared, inst)

        results = await asyncio.gather(*(attempt(n) for n in range(size)))
        rewards = [e["reward"] for e in results]
        return {"episodes": list(results), "group": group, "mixed": 0 < sum(rewards) < len(rewards)}


async def main(a) -> None:
    flyte.init_from_config(a.config, project=a.project, domain=a.domain)
    insts = tasks.load_tasks(a.instances or (tasks.SMOKE_IDS if a.smoke else tasks.DEFAULT_IDS))
    sampler, tokenizer = rollouts.make_sampler(a.sampler, base_url=a.base_url)
    cfg = {**(rollouts.SMOKE_CONFIG if a.smoke else rollouts.CONFIG), "group_size": a.group_size}
    # Every sandbox is a run of its own: one label on all of them groups this job's together.
    labels = {RUN_LABEL: a.run_id or f"{time.strftime('%m%d-%H%M')}-{uuid.uuid4().hex[:4]}"}
    print(f"sandboxes of this job are labelled {RUN_LABEL}={labels[RUN_LABEL]}")
    envs = UnionEnvironments(cache_testbeds=not a.no_cache, labels=labels, queue=a.queue)
    log = rollouts.EventLog(a.out)

    await asyncio.gather(*(envs.image(i) for i in insts))  # each instance's testbed: built, or reused
    job = Job(envs, sampler, tokenizer, cfg, log, rollout_workers=a.rollout_workers, grading_workers=a.grading_workers)
    groups: List[Dict[str, Any]] = await asyncio.gather(*(job.run_group(inst, g) for g, inst in enumerate(insts)))

    rollouts.print_table([e for g in groups for e in g["episodes"]])
    for g, inst in zip(groups, insts):
        rewards = [e["reward"] for e in g["episodes"]]
        mean = sum(rewards) / len(rewards)
        adv = ", ".join(f"{r - mean:+.2f}" for r in rewards)
        print(f"group {g['group']} {inst['instance_id']}: mixed={g['mixed']} (GRPO advantages {adv})")
    (Path(a.out) / "results.json").write_text(json.dumps({"labels": labels, "groups": groups}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="flyte config file (default: your current flyte config)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--domain", default="development")
    ap.add_argument("--smoke", action="store_true", help="2 instances, oracle sampler (no model needed)")
    ap.add_argument("--sampler", choices=["tinker", "openai", "oracle"], default="tinker")
    ap.add_argument("--base-url", default=None, help="for --sampler openai")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rollout-workers", type=int, default=16)
    ap.add_argument("--grading-workers", type=int, default=16)
    ap.add_argument("--no-cache", action="store_true", help="build testbeds afresh, publishing none")
    ap.add_argument("--out", default="swe-gym-runs")
    ap.add_argument("--queue", default="", help="run the sandboxes in this queue (default: the project's)")
    ap.add_argument("--run-id", default=None, help=f"value of the {RUN_LABEL} label on this job's sandboxes")
    args = ap.parse_args()
    if args.smoke:
        args.sampler = "oracle"
    asyncio.run(main(args))
