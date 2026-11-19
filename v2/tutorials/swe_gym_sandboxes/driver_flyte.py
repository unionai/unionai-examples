# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.4",
#    "flyteplugins-union>=0.23.0",
# ]
# main = "main"
# params = 'instance_ids=["getmoto__moto-7023","python__mypy-16781"]'
# ///
"""SWE-Gym rollouts as a Flyte 2 run: the same rollout code, made durable.

    main ─┬─ build_testbed(moto-7023)      cached: the testbed is built and sealed once, for every run
          ├─ build_testbed(mypy-16781)
          ├─ rollout(moto-7023, 0..3)      a group of 4: tasks with retries, one sandbox each,
          └─ rollout(mypy-16781, 0..3)     on copy-on-write forks of the testbed
                 └─ grade                  @flyte.trace: a retry after grading reuses its verdict

What Flyte adds over driver_laptop.py:
  - the testbed build is a cached task, so it happens once across runs, not once per process
  - each rollout is a task with retries=2: a rollout that dies (a node lost, a sandbox
    refused) is run again; InfrastructureError is what the rollout raises for those
  - grading is traced: if the attempt fails after grading, its retry replays the verdict
  - every run, attempt and sandbox is in the console, in order, with its logs

Run, from this directory:
  python driver_flyte.py --project P --smoke
  python driver_flyte.py --project P --sampler tinker

`--sampler tinker` reads TINKER_API_KEY from a Union secret of that name, given to the driver
tasks only: the sandboxes get no secrets and no network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import flyte

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rollouts  # noqa: E402
import swe_gym_tasks as tasks  # noqa: E402
from union_environments import Testbed, UnionEnvironments  # noqa: E402

from flyteplugins.union.sandbox import SandboxVolume  # noqa: E402

IN_TASK = bool(os.environ.get("FLYTE_INTERNAL_EXECUTION_ID"))


def _flag(name: str, default: str) -> str:
    """A flag's value, read before argparse: the image and secrets depend on the sampler."""
    argv = sys.argv
    return argv[argv.index(name) + 1] if name in argv[:-1] else default


SAMPLER = os.environ.get("SWE_RL_SAMPLER") or ("oracle" if "--smoke" in sys.argv else _flag("--sampler", "tinker"))


def _image() -> flyte.Image:
    img = flyte.Image.from_debian_base(name="swe-gym-rl", platform=("linux/amd64",)).with_pip_packages(
        "flyteplugins-union>=0.23.0"
    )
    if SAMPLER != "oracle":
        img = img.with_pip_packages("tinker", "transformers[chat-template]")
    return img


env = flyte.TaskEnvironment(
    name="swe_gym_rl",
    image=flyte.Image.from_debian_base(name="swe-gym-rl") if IN_TASK else _image(),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    env_vars={"SWE_RL_SAMPLER": SAMPLER},
    secrets=[flyte.Secret(key="TINKER_API_KEY", as_env_var="TINKER_API_KEY")] if SAMPLER == "tinker" else [],
)


@env.task(cache="auto", retries=1)
async def build_testbed(instance_json: str) -> Tuple[SandboxVolume, str]:
    """Builds the instance's testbed in a sandbox with network, and seals it: its disk,
    committed, and the tests that pass on it with the gold patch."""
    inst = json.loads(instance_json)
    envs = UnionEnvironments()
    tb = await envs.image(inst)
    return tb.volume, json.dumps({"instance_id": tb.instance_id, "pass_to_pass": tb.pass_to_pass})


class TracedEnvironments(UnionEnvironments):
    """Grading through a trace: its verdict is recorded, and a retry of the attempt replays it."""

    async def grade(self, inst, patch, directory):
        report = json.loads(await _grade(json.dumps(inst), patch))
        _write_report(Path(directory), report)
        return report

    async def _grade_now(self, inst, patch, directory):
        return await super().grade(inst, patch, directory)


def _write_report(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2))


_ENVS: List[TracedEnvironments] = []


@flyte.trace
async def _grade(instance_json: str, patch: str) -> str:
    report = await _ENVS[0]._grade_now(json.loads(instance_json), patch, "/tmp/swe-gym/grading")
    return json.dumps(report)


@env.task(retries=2)
async def rollout(instance_json: str, testbed: SandboxVolume, testbed_meta: str, n: int, base_url: str = "") -> str:
    """One attempt at one instance: a rollout in its own sandbox, then grading in another."""
    inst = json.loads(instance_json)
    meta = json.loads(testbed_meta)
    envs = TracedEnvironments(validate=False)
    envs.adopt(Testbed(inst["instance_id"], testbed, meta.get("pass_to_pass")))
    _ENVS[:] = [envs]
    sampler, tokenizer = rollouts.make_sampler(SAMPLER, base_url=base_url or None)
    cfg = rollouts.SMOKE_CONFIG if SAMPLER == "oracle" else rollouts.CONFIG
    log = rollouts.EventLog("/tmp/swe-gym")
    identity = {"phase": "rollout", "job": 0, "group": 0, "attempt": n}
    prepared = await rollouts.generate_rollout(inst, sampler, tokenizer, envs, cfg, log, identity)
    row = await rollouts.grade_rollout(prepared, inst, envs, log)
    print(json.dumps(row))
    return json.dumps(row)


@env.task
async def main(instance_ids: List[str], group_size: int = 4, base_url: str = "") -> str:
    insts = tasks.load_tasks(instance_ids)
    built = await asyncio.gather(*(build_testbed(json.dumps(i)) for i in insts))
    calls = [
        rollout(json.dumps(inst), volume, meta, n, base_url)
        for inst, (volume, meta) in zip(insts, built)
        for n in range(group_size)
    ]
    rows = [json.loads(r) for r in await asyncio.gather(*calls)]
    rollouts.print_table(rows)
    return json.dumps({"attempts": rows}, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="flyte config file (default: your current flyte config)")
    ap.add_argument("--project", default=None)
    ap.add_argument("--domain", default="development")
    ap.add_argument("--smoke", action="store_true", help="2 instances, oracle sampler (no model needed)")
    ap.add_argument("--sampler", choices=["tinker", "openai", "oracle"], default="tinker")
    ap.add_argument("--base-url", default="", help="for --sampler openai: e.g. http://vllm.my-ns.svc:8000/v1")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--group-size", type=int, default=4)
    a = ap.parse_args()
    ids = a.instances or (tasks.SMOKE_IDS if a.smoke else tasks.DEFAULT_IDS)
    # Bundle only this directory: the plugin comes from the image.
    flyte.init_from_config(a.config, root_dir=HERE)
    run = flyte.with_runcontext(project=a.project, domain=a.domain).run(
        main, instance_ids=ids, group_size=a.group_size, base_url=a.base_url
    )
    print(run.url)
    run.wait()
    print(run.outputs())
