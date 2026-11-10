# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# main = "batch"
# ///

"""Batch inference on National Compute's MI355X market that survives preemption.

National Compute sells whole 8-GPU nodes at a market price. When the price goes
above your limit, the node is taken back, and when it drops you get one again.

- The driver runs on a CPU node and splits the prompts into shards.
- Each shard runs on a reusable GPU environment where one replica is one full
  node. The replica starts a vLLM server once, and its weights stream from
  object storage straight into GPU memory.
- Shards are cached, and every sub-batch inside a shard is a trace, so a retry
  after a preemption replays the finished sub-batches.
- Replicas scale from one node up to `NC_MAX_NODES`. With more granted nodes
  the run goes faster, and with none it waits in the queue.

gpt-oss-120b runs as eight engines per node (`--tensor-parallel 1`). A model
that needs the whole node, such as Kimi K3, uses `--tensor-parallel 8`.

    uv run batch_inference.py --model <prefetched model dir> --num-prompts 2000
"""

import argparse
import asyncio
import html
import json
import os
import random
import subprocess
import sys
import time

import flyte
import flyte.report
from flyte.io import Dir, File

from common import (
    GPUS_PER_NODE,
    QUEUE,
    full_node_pod_template,
    prebuild_and_reexec,
    with_registry,
)

VLLM_ROCM_BASE = (
    "docker.io/vllm/vllm-openai-rocm:nightly-rocm100-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf"
)
SERVER_PORT = 8000
SERVED_NAME = "model"
SUB_BATCH = 128

# {{docs-fragment images}}
vllm_image_spec = (
    with_registry(flyte.Image.from_base(VLLM_ROCM_BASE), "nc-vllm-rocm")
    # The vLLM ROCm image ships Python 3.12; without this the build targets the local interpreter.
    .clone(python_version=(3, 12), extendable=True).with_pip_packages(
        "flyteplugins-vllm>=2.13.1", "unionai-reuse>=0.1.11", "openai", "async-lru"
    )
)
driver_image_spec = with_registry(
    flyte.Image.from_debian_base(python_version=(3, 12)), "nc-batch-driver"
)

# `__main__` builds the images without a queue and passes their URIs in, see
# common.prebuild_and_reexec.
vllm_image = os.getenv("NC_VLLM_IMAGE") or vllm_image_spec
driver_image = os.getenv("NC_DRIVER_IMAGE") or driver_image_spec
# {{/docs-fragment images}}

# {{docs-fragment envs}}
max_nodes = int(os.getenv("NC_MAX_NODES", "2"))

worker_env = flyte.TaskEnvironment(
    name="nc-vllm-worker",
    image=vllm_image,
    resources=flyte.Resources(cpu="32", memory="256Gi", shm="64Gi"),
    pod_template=full_node_pod_template(),
    queue=QUEUE,
    interruptible=True,
    # One replica per granted node. Several shards share a replica so the
    # server always has requests queued.
    reusable=flyte.ReusePolicy(
        replicas=(1, max_nodes), concurrency=4, idle_ttl=900, scaledown_ttl=300
    ),
)

driver_env = flyte.TaskEnvironment(
    name="nc-batch-driver",
    image=driver_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    queue=QUEUE,
    # The driver re-imports this module to submit shards; give it the same URIs.
    env_vars={k: v for k in ("NC_VLLM_IMAGE", "NC_DRIVER_IMAGE") if (v := os.getenv(k))},
    depends_on=[worker_env],
)
# {{/docs-fragment envs}}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    "parses an ISO 8601 duration string into seconds",
    "merges overlapping intervals in a list",
    "finds the longest palindromic substring",
    "implements an LRU cache with O(1) get and put",
    "validates a Sudoku board",
    "serializes and deserializes a binary tree",
    "computes the edit distance between two strings",
    "detects a cycle in a directed graph",
    "rate limits calls with a token bucket",
    "evaluates a reverse Polish notation expression",
    "groups anagrams from a list of words",
    "returns the k most frequent elements",
    "topologically sorts tasks with dependencies",
    "finds the median of a data stream",
    "decodes a run-length encoded string",
    "checks whether a string has balanced brackets",
]
LANGUAGES = ["Python", "TypeScript", "Go", "Rust"]
STYLES = [
    "Include unit tests.",
    "Explain the time and space complexity.",
    "Handle invalid input explicitly.",
    "Keep it under 40 lines.",
]


def make_prompts(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "id": i,
            "prompt": f"Write a {rng.choice(LANGUAGES)} function that {rng.choice(TASKS)}. {rng.choice(STYLES)}",
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# vLLM server, one per replica
# ─────────────────────────────────────────────────────────────────────────────

_server: dict = {}
_server_lock = asyncio.Lock()


async def ensure_server(model_path: str, tensor_parallel: int, extra_args: str) -> float:
    """Start vLLM once per replica and return how long the load took."""
    async with _server_lock:
        if _server.get("proc") is not None and _server["proc"].poll() is None:
            return _server["load_seconds"]

        import httpx
        from flyteplugins.vllm.model_streamer import engine_args

        start = time.time()
        args = await engine_args(model_path)
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args["model"],
            "--served-model-name",
            SERVED_NAME,
            "--load-format",
            args["load_format"],
            "--model-loader-extra-config",
            json.dumps(args["model_loader_extra_config"]),
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--data-parallel-size",
            str(GPUS_PER_NODE // tensor_parallel),
            "--port",
            str(SERVER_PORT),
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.92",
            *extra_args.split(),
        ]
        print("starting vLLM:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        async with httpx.AsyncClient() as client:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {proc.returncode} while loading"
                    )
                try:
                    if (
                        await client.get(f"http://localhost:{SERVER_PORT}/health", timeout=5)
                    ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)

        _server.update(proc=proc, load_seconds=time.time() - start)
        print(f"vLLM ready in {_server['load_seconds']:.0f}s", flush=True)
        return _server["load_seconds"]


# {{docs-fragment generate}}
@flyte.trace
async def generate(prompts: list[dict], max_tokens: int) -> list[dict]:
    """One sub-batch. Traced, so a retried shard replays it instead of rerunning it."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"http://localhost:{SERVER_PORT}/v1", api_key="none")

    async def one(p: dict) -> dict:
        resp = await client.chat.completions.create(
            model=SERVED_NAME,
            messages=[{"role": "user", "content": p["prompt"]}],
            max_tokens=max_tokens,
        )
        return {
            "id": p["id"],
            "completion": resp.choices[0].message.content,
            "tokens": resp.usage.completion_tokens,
        }

    return list(await asyncio.gather(*(one(p) for p in prompts)))


# {{/docs-fragment generate}}


# {{docs-fragment shard}}
@worker_env.task(cache="auto", retries=10)
async def infer_shard(
    model: Dir,
    shard: File,
    tensor_parallel: int = 1,
    max_tokens: int = 1024,
    extra_args: str = "",
    crash: bool = False,
) -> File:
    load_seconds = await ensure_server(model.path, tensor_parallel, extra_args)

    async with shard.open("rb") as f:
        prompts = json.loads(await f.read())
    batches = [prompts[i : i + SUB_BATCH] for i in range(0, len(prompts), SUB_BATCH)]

    results: list[dict] = []
    for i, batch in enumerate(batches):
        results.extend(await generate(batch, max_tokens))
        # Demo switch: on the first attempt, kill the replica halfway through.
        # The retry lands on a replica (this one restarted, or another node),
        # replays the finished sub-batches from their traces and carries on.
        if crash and flyte.ctx().attempt_number == 0 and i == (len(batches) - 1) // 2:
            print(f"crashing after sub-batch {i}", flush=True)
            os._exit(137)

    out = File.new_remote()
    async with out.open("wb") as f:
        await f.write(json.dumps({"load_seconds": load_seconds, "results": results}).encode())
    return out


# {{/docs-fragment shard}}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def _render(
    done: int, total: int, prompts_done: int, tokens: int, start: float, loads: list[float]
) -> str:
    elapsed = max(time.time() - start, 1e-6)
    cards = [
        ("Shards", f"{done} / {total}"),
        ("Prompts", f"{prompts_done:,}"),
        ("Output tokens", f"{tokens:,}"),
        ("Tokens / s", f"{tokens / elapsed:,.0f}"),
        ("Elapsed", f"{elapsed / 60:.1f} min"),
        ("Model load (median)", f"{sorted(loads)[len(loads) // 2]:.0f} s" if loads else "-"),
    ]
    items = "".join(
        f"<div style='padding:12px 16px;border:1px solid #ddd;border-radius:8px'>"
        f"<div style='color:#666;font-size:12px'>{html.escape(k)}</div>"
        f"<div style='font-size:22px'>{html.escape(v)}</div></div>"
        for k, v in cards
    )
    return f"<h2>National Compute batch inference</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>{items}</div>"


@driver_env.task(report=True)
async def batch(
    model: Dir,
    num_prompts: int = 2000,
    shard_size: int = 256,
    tensor_parallel: int = 1,
    max_tokens: int = 1024,
    extra_args: str = "",
    crash_shard: int = -1,
) -> Dir:
    prompts = make_prompts(num_prompts)
    shards = []
    for i in range(0, len(prompts), shard_size):
        f = File.new_remote()
        async with f.open("wb") as fh:
            await fh.write(json.dumps(prompts[i : i + shard_size]).encode())
        shards.append(f)

    start = time.time()
    state = {"done": 0, "prompts": 0, "tokens": 0, "loads": []}
    outputs: list[File] = []

    async def run_one(i: int, shard: File) -> None:
        out = await infer_shard(
            model, shard, tensor_parallel, max_tokens, extra_args, crash=(i == crash_shard)
        )
        async with out.open("rb") as fh:
            data = json.loads(await fh.read())
        outputs.append(out)
        state["done"] += 1
        state["prompts"] += len(data["results"])
        state["tokens"] += sum(r["tokens"] or 0 for r in data["results"])
        state["loads"].append(data["load_seconds"])
        await flyte.report.replace.aio(
            _render(
                state["done"],
                len(shards),
                state["prompts"],
                state["tokens"],
                start,
                state["loads"],
            ),
            do_flush=True,
        )

    await flyte.report.replace.aio(_render(0, len(shards), 0, 0, start, []), do_flush=True)
    await asyncio.gather(*(run_one(i, s) for i, s in enumerate(shards)))

    local = "/tmp/results"
    os.makedirs(local, exist_ok=True)
    for j, out in enumerate(outputs):
        await out.download(os.path.join(local, f"shard-{j:05d}.json"))
    return await Dir.from_local(local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Remote path of the prefetched model directory"
    )
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--crash-shard", type=int, default=-1)
    a = parser.parse_args()

    prebuild_and_reexec(
        {"NC_VLLM_IMAGE": vllm_image_spec, "NC_DRIVER_IMAGE": driver_image_spec}
    )
    flyte.init_from_config()
    run = flyte.with_runcontext(queue=QUEUE).run(
        batch,
        model=Dir.from_existing_remote(a.model),
        num_prompts=a.num_prompts,
        shard_size=a.shard_size,
        tensor_parallel=a.tensor_parallel,
        max_tokens=a.max_tokens,
        extra_args=a.extra_args,
        crash_shard=a.crash_shard,
    )
    print(run.name, run.url)
