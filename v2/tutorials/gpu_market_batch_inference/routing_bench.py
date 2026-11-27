# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
# ]
# ///

"""Compare prefix routing with spreading prompts evenly, on two or more Kimi K3 apps.

One CPU task sends the prompts the way batch_app.py's shards would: each
node's prompts in `streams` contiguous chunks, each chunk in sub-batches of
`sub_batch` requests at a time. It does this once with prefix routing and once
with spread routing, on different documents so the second run cannot reuse the
first run's cache, and reports wall time and vLLM's cache counters per node.

    uv run routing_bench.py --endpoint <url a> <url b> --docs 600 --questions 8
"""

import argparse
import asyncio
import json
import os
import time

import flyte

from batch_app import (
    API_KEY,
    SERVED_NAME,
    cache_counters,
    image,
    make_document_prompts,
    render_prompt,
    route,
    wait_for_app,
    warm_up,
)

env = flyte.TaskEnvironment(
    name="kimi-routing-bench",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    secrets=API_KEY,
)


async def send_node(
    endpoint: str, prompts: list[dict], streams: int, sub_batch: int, max_tokens: int, failures: list[str]
) -> list[float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"{endpoint}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=900, max_retries=8)

    async def one(p: dict) -> float:
        start = time.perf_counter()
        for attempt in range(6):
            try:
                await client.chat.completions.create(
                    model=SERVED_NAME, messages=[{"role": "user", "content": render_prompt(p)}], max_tokens=max_tokens
                )
                return time.perf_counter() - start
            except Exception as e:
                # The server crashed or restarted: wait until it answers, then send again.
                failures.append(f"{endpoint} {type(e).__name__}")
                if attempt == 5:
                    raise
                await wait_for_app(endpoint)
        raise AssertionError("unreachable")

    async def stream(chunk: list[dict]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(chunk), sub_batch):
            out.extend(await asyncio.gather(*(one(p) for p in chunk[i : i + sub_batch])))
        return out

    size = -(-len(prompts) // streams)
    chunks = [prompts[i : i + size] for i in range(0, len(prompts), size)]
    return [x for xs in await asyncio.gather(*(stream(c) for c in chunks)) for x in xs]


@env.task
async def bench(
    endpoints: list[str],
    docs: int = 600,
    questions: int = 8,
    streams: int = 4,
    sub_batch: int = 64,
    max_tokens: int = 64,
    routings: list[str] = ["prefix", "spread"],
) -> str:
    await asyncio.gather(*(warm_up(e) for e in endpoints))
    rows = []
    for k, routing in enumerate(routings):
        prompts = make_document_prompts(docs, questions, seed=100 + k)
        per_node = route(prompts, len(endpoints), routing, seed=100 + k)
        before = {e: await cache_counters(e) for e in endpoints}
        start = time.time()
        node_times = {}
        failures: list[str] = []

        async def run(e: str, ps: list[dict]) -> None:
            t0 = time.time()
            lat = await send_node(e, ps, streams, sub_batch, max_tokens, failures)
            node_times[e] = {"wall_s": round(time.time() - t0, 1), "p50_s": round(sorted(lat)[len(lat) // 2], 2)}

        await asyncio.gather(*(run(e, ps) for e, ps in zip(endpoints, per_node)))
        elapsed = time.time() - start
        nodes = []
        for e, ps in zip(endpoints, per_node):
            after = await cache_counters(e)
            d = {k2: after[k2] - before[e].get(k2, 0.0) for k2 in after if isinstance(after[k2], float)}
            hits, queries = d.get("vllm:prefix_cache_hits_total", 0.0), d.get("vllm:prefix_cache_queries_total", 0.0)
            nodes.append(
                {
                    "endpoint": e,
                    "prompts": len(ps),
                    "docs": len({p["prefix_key"] for p in ps}),
                    **node_times[e],
                    "prompt_tokens": d.get("vllm:prompt_tokens_total"),
                    "gpu_prefix_hit_rate": round(hits / queries, 3) if queries else None,
                    "counters": {k2: round(v, 1) for k2, v in d.items() if v and ("offload" in k2 or "prefix" in k2 or "kv_" in k2)},
                }
            )
        row = {"routing": routing, "prompts": len(prompts), "elapsed_s": round(elapsed, 1),
               "prompts_per_s": round(len(prompts) / elapsed, 2), "failed_requests": len(failures),
               "failed_by_node": {e: sum(f.startswith(e) for f in failures) for e in endpoints}, "nodes": nodes}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return json.dumps(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, nargs="+")
    parser.add_argument("--docs", type=int, default=600)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--routings", default="prefix,spread")
    a = parser.parse_args()
    flyte.init_from_config()
    run = flyte.run(
        bench, endpoints=[e.rstrip("/") for e in a.endpoint], docs=a.docs, questions=a.questions,
        routings=a.routings.split(","),
    )
    print(run.name, run.url)
