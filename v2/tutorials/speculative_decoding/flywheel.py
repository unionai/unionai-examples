# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# ///
"""Keep the draft head matched to production traffic.

Every night, `retrain`:

1. Collects the traces that `agent_job` runs produced in the last day.
2. Holds out the most recent 10% as an evaluation set.
3. Fine-tunes the current draft head on the rest, plus replayed general chat data so the
   head does not forget how to draft outside the agent workload.
4. Benchmarks the candidate against the current head on the held-out traces.
5. Returns the candidate only if its acceptance length beats the current head's.

    python flywheel.py              # register the nightly trigger
    python flywheel.py --now        # run one retraining cycle immediately
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

import flyte
import flyte.remote as remote
from flyte.io import Dir, File

from benchmark import ServerConfig, benchmark_server
from report import gate_report
from config import bench_env, cpu_image, data_env, generate_env, train_env
from data import prepare_prompts, regenerate_chat
from train import train_drafter

flywheel_env = flyte.TaskEnvironment(
    name="specdec-flywheel",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    depends_on=[data_env, generate_env, train_env, bench_env],
)


async def _output_uri(run: remote.Run) -> str:
    """The storage URI of a run's single file or directory output."""
    (literal,) = (await run.output_literals.aio()).values()
    return literal.scalar.blob.uri


# {{docs-fragment collect}}
async def collect_traces(since: datetime, until: datetime) -> list[dict]:
    """Read the trace files returned by every successful `agent_job` run in the window."""
    traces = []
    runs = remote.Run.listall.aio(
        in_phase=("succeeded",),
        task_name="specdec-agent.agent_job",
        created_at=remote.TimeFilter(after=since, before=until),
        limit=500,
    )
    async for run in runs:
        trace_file = File.from_existing_remote(await _output_uri(run))
        with open(await trace_file.download()) as f:
            traces.extend(json.loads(line) for line in f)
    return traces


async def latest_drafter() -> Dir:
    """The draft head currently in production: the newest retrain or pipeline output."""
    for task_name in ("specdec-flywheel.retrain", "specdec-driver.main"):
        async for run in remote.Run.listall.aio(
            in_phase=("succeeded",), task_name=task_name, sort_by=("created_at", "desc"), limit=1
        ):
            return Dir.from_existing_remote(await _output_uri(run))
    raise RuntimeError("No trained draft head yet. Run pipeline.py first.")
# {{/docs-fragment collect}}


def _eval_prompts(traces: list[dict]) -> list[dict]:
    """Every assistant turn in the held-out traces becomes one benchmark prompt."""
    prompts = []
    for trace in traces:
        messages = trace["messages"]
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                prompts.append({"messages": messages[:i], "tools": trace["tools"]})
    return prompts


async def _write_jsonl(rows: list[dict], path: str) -> File:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return await File.from_local(path)


# {{docs-fragment retrain}}
@flywheel_env.task(
    report=True,
    triggers=flyte.Trigger(
        name="nightly-drafter-retrain",
        automation=flyte.Cron("0 3 * * *"),
        inputs={"window_end": flyte.TriggerTime},
    ),
)
async def retrain(
    window_end: datetime,
    window_hours: int = 24,
    min_traces: int = 200,
    min_gain: float = 0.02,
    max_steps: int | None = None,
) -> Dir:
    """Fine-tune the production draft head on yesterday's traces; promote it if it wins."""
    since = window_end - timedelta(hours=window_hours)
    window = f"Traces from {since:%Y-%m-%d %H:%M} to {window_end:%Y-%m-%d %H:%M} UTC"
    current = await latest_drafter()
    traces = await collect_traces(since, window_end)

    async def finish(drafter: Dir, promoted: bool, reason: str, **details) -> Dir:
        await flyte.report.replace.aio(
            gate_report(window, len(traces), promoted=promoted, reason=reason, **details)
        )
        await flyte.report.flush.aio()
        return drafter

    if len(traces) < min_traces:
        return await finish(
            current, False, f"only {len(traces)} traces, below min_traces={min_traces}",
            n_train=0, n_eval_prompts=0, acceptance=None,
        )

    cut = int(len(traces) * 0.9)
    train_file = await _write_jsonl(traces[:cut], "/tmp/train-traces.jsonl")

    # Replay general chat so the head keeps drafting well outside the agent workload.
    # Same inputs as the pipeline's first shard, so this is a cache hit, not a GPU job.
    shards = await prepare_prompts(num_chat=4000, num_agent=4000, num_shards=4)
    replay = (await regenerate_chat(shards.chat[0])).data

    candidate = await train_drafter(
        data=[train_file, replay], warm_start=current, epochs=1, max_steps=max_steps
    )

    # Gate: benchmark both heads on the most recent, held-out traffic.
    eval_prompts = _eval_prompts(traces[cut:])
    eval_path = "/tmp/eval-prompts.json"
    with open(eval_path, "w") as f:
        json.dump({"agent-recent": eval_prompts}, f)
    prompts = await File.from_local(eval_path)
    with flyte.group("gate"):
        results = await asyncio.gather(
            *(
                benchmark_server(ServerConfig(name=n, method="eagle3", drafter=d), prompts, [1])
                for n, d in (("current", current), ("candidate", candidate))
            )
        )
    al = {}
    for f in results:
        with open(await f.download()) as fh:
            r = json.load(fh)
        al[r["config"]] = r["workloads"]["agent-recent"]["1"]["acceptance_length"]

    details = dict(n_train=cut, n_eval_prompts=len(eval_prompts), acceptance=al)
    if al["candidate"] >= al["current"] * (1 + min_gain):
        return await finish(candidate, True, f"acceptance length improved by at least {min_gain:.0%}", **details)
    return await finish(current, False, f"gain below the {min_gain:.0%} threshold", **details)
# {{/docs-fragment retrain}}


if __name__ == "__main__":
    flyte.init_from_config()
    if "--now" in sys.argv:
        run = flyte.run(retrain, window_end=datetime.now(timezone.utc))
        print(run.url)
    else:
        # Deploying the environment registers its trigger.
        flyte.deploy(flywheel_env)
