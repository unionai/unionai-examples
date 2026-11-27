# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
# ]
# main = "batch"
# ///

"""Batch inference against Kimi K3 apps, with shards that survive failures.

`kimi_app.py` serves the model on MI355X nodes. This script runs the batch on
small CPU tasks that send requests to those apps:

- The driver splits the prompts into shards and runs one task per shard.
- Shards are cached and retried. Each sub-batch of a shard is a `@flyte.trace`,
  so a retried shard replays the sub-batches it already finished.
- The driver waits until every app answers before it starts the shards, so the
  batch can be started while the apps are still waiting for nodes or loading
  the model.
- A shard whose app stops answering fails within minutes instead of waiting.
  Its retry comes back with growing pauses (30 s, doubling, up to 5 min), which
  covers the market replacing a node, and the CPU is free in between.

With more than one app, one per node, the driver decides which node gets each
prompt. Prompts that share a prefix, such as questions about the same
document, go to the same node, so the node computes that prefix once and reads
it from its cache for the rest. Each node's prompts are sorted so matching
prefixes run back to back, before the prefix leaves the GPU cache. A plain
hash would pile a popular prefix onto one node, so a node that already holds
its share of the prompts passes the next prefix on to the following node.

`--routing spread` instead shuffles the prompts and deals them out evenly,
which is what a load balancer in front of several replicas does. It is the
baseline for prefix routing.

    uv run batch_app.py --endpoint <app url> [<app url> ...] --num-prompts 1024
    uv run batch_app.py --endpoint <url a> <url b> --workload documents --docs 600 --questions 8
"""

import argparse
import asyncio
import html
import json
import os
import random
import time
from datetime import timedelta
from contextvars import ContextVar

import flyte
import flyte.report
from flyte.io import File

SERVED_NAME = "kimi-k3"
SUB_BATCH = 64
API_KEY = flyte.Secret(key="kimi-vllm-api-key", as_env_var="VLLM_API_KEY")

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("openai", "httpx")

shard_env = flyte.TaskEnvironment(
    name="kimi-shard",
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    secrets=API_KEY,
)
driver_env = flyte.TaskEnvironment(
    name="kimi-batch",
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    # The driver warms up the apps before timing the batch.
    secrets=API_KEY,
    depends_on=[shard_env],
)

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


DOC_TOPICS = ["caching", "retries", "scheduling", "storage", "networking", "billing", "security", "logging"]


def make_document(seed: int, target_words: int = 6000) -> str:
    """A synthetic reference document of about 8,700 tokens."""
    rng = random.Random(seed)
    lines, words, i = [], 0, 0
    while words < target_words:
        t = rng.choice(DOC_TOPICS)
        line = (
            f"Fact {i} ({t}): component {rng.randint(100, 999)} uses a {t} policy with limit "
            f"{rng.randint(1, 500)} and a timeout of {rng.randint(1, 60)} seconds, owned by team {rng.randint(1, 40)}."
        )
        lines.append(line)
        words += len(line.split())
        i += 1
    return f"Reference document, version {seed}.\n" + "\n".join(lines)


def make_document_prompts(docs: int, questions: int, seed: int = 0) -> list[dict]:
    """Several questions about each document. The document is the shared prefix.

    Prompts hold the document's seed rather than its text, and the shard
    rebuilds the text, which keeps the shard files small.
    """
    rng = random.Random(seed)
    prompts = []
    for d in range(docs):
        doc_seed = seed * 1_000_003 + d
        for q in rng.sample(range(200), questions):
            prompts.append(
                {
                    "id": len(prompts),
                    "prefix_key": f"doc-{doc_seed}",
                    "doc_seed": doc_seed,
                    "prompt": f"Which team owns fact {q}, and what is its timeout? Answer in one sentence.",
                }
            )
    return prompts


def render_prompt(p: dict) -> str:
    if "doc_seed" in p:
        return f"{make_document(p['doc_seed'])}\n\nQuestion: {p['prompt']}"
    return p["prompt"]


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────


def _hash(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def route(prompts: list[dict], n: int, routing: str, seed: int = 0) -> list[list[dict]]:
    """Split the prompts into one ordered list per node.

    prefix: consistent hashing with bounded load. Each prefix group starts at
    its point on a hash ring and goes to the first node that still has room
    for it. A node has room up to an even share plus one group, so the nodes
    finish at about the same time. Groups stay whole and in order.

    spread: shuffle, then deal the prompts out evenly.
    """
    if routing == "spread":
        shuffled = list(prompts)
        random.Random(seed).shuffle(shuffled)
        return [shuffled[i::n] for i in range(n)]

    groups: dict[str, list[dict]] = {}
    for p in prompts:
        groups.setdefault(p.get("prefix_key") or p["prompt"][:256], []).append(p)
    ring = sorted((_hash(f"node-{i}-{v}"), i) for i in range(n) for v in range(64))
    cap = len(prompts) / n + max(len(g) for g in groups.values())
    loads = [0] * n
    nodes: list[list[dict]] = [[] for _ in range(n)]
    # Large groups first, so the small ones fill the gaps they leave.
    for key, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        h = _hash(key)
        start = next((j for j, (pos, _) in enumerate(ring) if pos >= h), 0)
        order = list(dict.fromkeys(ring[(start + j) % len(ring)][1] for j in range(len(ring))))
        target = next((i for i in order if loads[i] + len(group) <= cap), min(range(n), key=loads.__getitem__))
        loads[target] += len(group)
        nodes[target].append((key, group))
    # Within a node, matching prefixes run back to back.
    return [[p for _, group in sorted(node) for p in group] for node in nodes]


# ─────────────────────────────────────────────────────────────────────────────
# Shards
# ─────────────────────────────────────────────────────────────────────────────


async def wait_for_app(endpoint: str, timeout_s: float = 3 * 3600) -> None:
    """Wait until the app serves the model. Covers node grants and model load."""
    import httpx

    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                r = await client.get(f"{endpoint}/v1/models", headers=headers)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise TimeoutError(f"{endpoint} did not become ready in {timeout_s / 60:.0f} minutes")
            await asyncio.sleep(20)


# Counts the sub-batches a shard invocation actually generates. A replayed trace
# skips the function body, so replays don't add to it.
_generated: ContextVar[list[int] | None] = ContextVar("_generated", default=None)


@flyte.trace
async def generate(endpoint: str, prompts: list[dict], max_tokens: int) -> list[dict]:
    """One sub-batch. Traced, so a retried shard replays it instead of rerunning it."""
    from openai import AsyncOpenAI

    counter = _generated.get()
    if counter is not None:
        counter.append(len(prompts))

    # A dead server drops the connection and the client retries. A stuck one is
    # stopped by the app's watchdog within minutes, so a request never waits long.
    client = AsyncOpenAI(
        base_url=f"{endpoint}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=900, max_retries=8
    )

    async def one(p: dict) -> dict:
        resp = await client.chat.completions.create(
            model=SERVED_NAME,
            messages=[{"role": "user", "content": render_prompt(p)}],
            max_tokens=max_tokens,
        )
        return {
            "id": p["id"],
            "completion": resp.choices[0].message.content,
            "tokens": resp.usage.completion_tokens,
            "prompt_tokens": resp.usage.prompt_tokens,
        }

    return list(await asyncio.gather(*(one(p) for p in prompts)))


async def count_attempt() -> int:
    """Count this shard's attempts in the run's storage, keyed by its action name."""
    import flyte.storage as storage

    ctx = flyte.ctx()
    path = f"{ctx.run_base_dir.rstrip('/')}/attempts/{ctx.action.name}"
    count = 0
    if await storage.exists(path):
        count = int(b"".join([chunk async for chunk in storage.get_stream(path)]))
    count += 1
    if "://" not in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    await storage.put_stream(str(count).encode(), to_path=path)
    return count


# About 85 minutes of retries in all: enough for the market to grant a new node
# and the app to load the model on it.
SHARD_RETRIES = flyte.RetryStrategy(
    count=20, backoff=flyte.Backoff(base=timedelta(seconds=30), factor=2.0, cap=timedelta(minutes=5))
)
# How long a shard waits for its app before giving up its CPU and retrying later.
SHARD_APP_WAIT_S = 120


@shard_env.task(cache="auto", retries=SHARD_RETRIES)
async def infer_shard(endpoint: str, shard: File, max_tokens: int = 512, crash: bool = False) -> File:
    attempt = await count_attempt()
    await wait_for_app(endpoint, timeout_s=SHARD_APP_WAIT_S)

    async with shard.open("rb") as f:
        prompts = json.loads(bytes(await f.read()))
    batches = [prompts[i : i + SUB_BATCH] for i in range(0, len(prompts), SUB_BATCH)]

    generated: list[int] = []
    _generated.set(generated)
    results: list[dict] = []
    start = time.time()
    for i, batch in enumerate(batches):
        results.extend(await generate(endpoint, batch, max_tokens))
        # Demo switch: on the first attempt, kill this shard halfway through.
        # The retry replays the finished sub-batches from their traces.
        if crash and attempt == 1 and i == (len(batches) - 1) // 2:
            print(f"crashing after sub-batch {i}", flush=True)
            os._exit(137)

    summary = {
        "attempt": attempt,
        "batches": len(batches),
        "replayed": len(batches) - len(generated),
        "seconds": time.time() - start,
        "results": results,
    }
    out = File.new_remote()
    async with out.open("wb") as f:
        await f.write(json.dumps(summary).encode())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def _render(total: int, start: float, shards: dict[int, dict]) -> str:
    elapsed = max(time.time() - start, 1e-6)
    tokens = sum(r["tokens"] or 0 for s in shards.values() for r in s["results"])
    retried = {i: s for i, s in shards.items() if s["attempt"] > 1}
    cards = [
        ("Shards done", f"{len(shards)} / {total}"),
        ("Prompts", f"{sum(len(s['results']) for s in shards.values()):,}"),
        ("Output tokens", f"{tokens:,}"),
        ("Tokens / s", f"{tokens / elapsed:,.0f}"),
        ("Elapsed", f"{elapsed / 60:.1f} min"),
        ("Retried shards", str(len(retried))),
        ("Batches replayed", str(sum(s["replayed"] for s in shards.values()))),
    ]
    items = "".join(
        f"<div style='padding:12px 16px;border:1px solid #ddd;border-radius:8px'>"
        f"<div style='color:#666;font-size:12px'>{html.escape(k)}</div>"
        f"<div style='font-size:22px'>{html.escape(v)}</div></div>"
        for k, v in cards
    )
    out = f"<h2>Kimi K3 batch inference</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>{items}</div>"
    if retried:
        rows = "".join(
            f"<tr><td>{i}</td><td>{s['attempt']}</td><td>{s['replayed']} of {s['batches']}</td></tr>"
            for i, s in sorted(retried.items())
        )
        out += (
            "<h3>Retried shards</h3><table><tr><th>Shard</th><th>Attempts</th>"
            f"<th>Batches replayed</th></tr>{rows}</table>"
        )
    return out


async def cache_counters(endpoint: str) -> dict[str, float]:
    """vLLM's prefix cache, prompt and generation counters, summed over labels."""
    import httpx

    keep = ("prefix_cache", "prompt_tokens", "generation_tokens", "offload", "kv_")
    out: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            text = (await client.get(f"{endpoint}/metrics")).text
    except httpx.HTTPError as e:
        return {"error": str(e)}  # type: ignore[dict-item]
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(k in name for k in keep) and not name.endswith(("_bucket", "_created")):
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


async def warm_up(endpoint: str) -> None:
    """A few requests so first-request kernel compilation stays out of the timing."""
    from openai import AsyncOpenAI

    await wait_for_app(endpoint)
    client = AsyncOpenAI(base_url=f"{endpoint}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=1800, max_retries=5)
    await asyncio.gather(
        *(
            client.chat.completions.create(
                model=SERVED_NAME,
                messages=[{"role": "user", "content": f"{make_document(-1 - i)}\n\nQuestion: which team owns fact 3?"}],
                max_tokens=8,
            )
            for i in range(8)
        )
    )


@driver_env.task(report=True)
async def batch(
    endpoints: list[str],
    num_prompts: int = 1024,
    shard_size: int = 256,
    max_tokens: int = 512,
    crash_shard: int = -1,
    workload: str = "coding",
    docs: int = 600,
    questions: int = 8,
    routing: str = "prefix",
    seed: int = 0,
) -> tuple[list[File], str]:
    if workload == "documents":
        prompts = make_document_prompts(docs, questions, seed)
    else:
        prompts = make_prompts(num_prompts, seed)

    # One ordered prompt list per node, cut into shards that keep that order.
    shards: list[tuple[str, File]] = []
    per_node: dict[str, int] = {}
    for endpoint, node_prompts in zip(endpoints, route(prompts, len(endpoints), routing, seed=seed)):
        per_node[endpoint] = len(node_prompts)
        for i in range(0, len(node_prompts), shard_size):
            f = File.new_remote()
            async with f.open("wb") as fh:
                await fh.write(json.dumps(node_prompts[i : i + shard_size]).encode())
            shards.append((endpoint, f))

    await asyncio.gather(*(warm_up(e) for e in endpoints))
    before = {e: await cache_counters(e) for e in endpoints}
    start = time.time()
    finished: dict[int, dict] = {}
    outputs: list[File] = []

    async def run_one(i: int, endpoint: str, shard: File) -> None:
        out = await infer_shard(endpoint, shard, max_tokens, crash=(i == crash_shard))
        async with out.open("rb") as fh:
            finished[i] = json.loads(bytes(await fh.read()))
        outputs.append(out)
        await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)

    await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)
    await asyncio.gather(*(run_one(i, e, s) for i, (e, s) in enumerate(shards)))
    elapsed = time.time() - start
    await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)

    nodes = []
    for e in endpoints:
        after = await cache_counters(e)
        d = {k: after[k] - before[e].get(k, 0.0) for k in after if isinstance(after[k], float)}
        hits, queries = d.get("vllm:prefix_cache_hits_total", 0.0), d.get("vllm:prefix_cache_queries_total", 0.0)
        nodes.append(
            {
                "endpoint": e,
                "prompts": per_node[e],
                "prompt_tokens": d.get("vllm:prompt_tokens_total"),
                "generation_tokens": d.get("vllm:generation_tokens_total"),
                "gpu_prefix_hit_rate": round(hits / queries, 3) if queries else None,
                "counters": {k: round(v, 1) for k, v in d.items() if v},
            }
        )
    results = [r for s in finished.values() for r in s["results"]]
    summary = {
        "workload": workload,
        "routing": routing,
        "prompts": len(results),
        "elapsed_s": round(elapsed, 1),
        "prompts_per_s": round(len(results) / elapsed, 2),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in results),
        "output_tokens": sum(r["tokens"] or 0 for r in results),
        "nodes": nodes,
    }
    print(json.dumps(summary, indent=1), flush=True)
    return outputs, json.dumps(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, nargs="+", help="Public URL of each Kimi K3 app, one per node")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--crash-shard", type=int, default=-1)
    parser.add_argument("--workload", default="coding", choices=["coding", "documents"])
    parser.add_argument("--docs", type=int, default=600, help="Documents, for --workload documents")
    parser.add_argument("--questions", type=int, default=8, help="Questions per document")
    parser.add_argument("--routing", default="prefix", choices=["prefix", "spread"])
    parser.add_argument("--seed", type=int, default=0, help="Changes the prompts, so runs do not share a cache")
    a = parser.parse_args()

    flyte.init_from_config()
    run = flyte.run(
        batch,
        endpoints=[e.rstrip("/") for e in a.endpoint],
        num_prompts=a.num_prompts,
        shard_size=a.shard_size,
        max_tokens=a.max_tokens,
        crash_shard=a.crash_shard,
        workload=a.workload,
        docs=a.docs,
        questions=a.questions,
        routing=a.routing,
        seed=a.seed,
    )
    print(run.name, run.url)
