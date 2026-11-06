"""Benchmark speculative decoding configurations, one GPU per configuration.

Each configuration gets its own vLLM server on identical hardware and replays the same
prompts. The benchmark records three things:

- Speed: per-request decode speed, aggregate throughput, TTFT, and time per output token
  across a concurrency sweep.
- Draft quality: mean acceptance length from vLLM's speculative decoding counters.
- Output: the greedy completions, so the report can verify the target's outputs are
  unchanged and that task accuracy (GSM8K) is preserved.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import flyte
from flyte.io import Dir, File

from config import NUM_SPECULATIVE_TOKENS, PUBLIC_DRAFTER, TARGET_MODEL, bench_env, data_env
from report import server_report
from vllm_server import TOOL_ARGS, read_counters, start_server
from workloads import NO_THINKING, load_agent_tasks, load_gsm8k, load_mt_bench


# {{docs-fragment configs}}
@dataclass
class ServerConfig:
    """One way of serving the target model."""

    name: str
    method: str = ""  # "", "ngram", "suffix", or "eagle3"
    num_speculative_tokens: int = NUM_SPECULATIVE_TOKENS
    drafter_hf: str = ""  # A Hugging Face draft head...
    drafter: Dir | None = None  # ...or one trained by this pipeline.

    def speculative_config(self, drafter_path: str | None) -> dict | None:
        if not self.method:
            return None
        config = {"method": self.method, "num_speculative_tokens": self.num_speculative_tokens}
        if self.method == "ngram":
            config |= {"prompt_lookup_min": 2, "prompt_lookup_max": 5}
        if self.method == "eagle3":
            config["model"] = drafter_path or self.drafter_hf
        return config


def default_configs(trained: Dir | None = None) -> list[ServerConfig]:
    """No speculation, two draft-free methods, the public head, and the fine-tuned head."""
    configs = [
        ServerConfig(name="baseline"),
        ServerConfig(name="ngram", method="ngram", num_speculative_tokens=4),
        ServerConfig(name="suffix", method="suffix", num_speculative_tokens=16),
        ServerConfig(name="eagle3-public", method="eagle3", drafter_hf=PUBLIC_DRAFTER),
    ]
    if trained is not None:
        configs.append(ServerConfig(name="eagle3-finetuned", method="eagle3", drafter=trained))
    return configs
# {{/docs-fragment configs}}


# {{docs-fragment prompts}}
@data_env.task(cache="auto")
async def prepare_benchmark_prompts(n_per_workload: int = 78) -> File:
    """Held-out prompts for three workloads. None of them appear in the training data.

    The benchmark splits each workload across its concurrency levels, so 78 prompts give 26
    unseen prompts per level.
    """
    agent = load_agent_tasks("heldout", n_per_workload)
    prompts = {
        "chat": [{"messages": m} for m in load_mt_bench()[:n_per_workload]],
        "agent": [
            {
                "messages": [
                    {"role": "system", "content": "Complete the request by calling the provided tools."},
                    {"role": "user", "content": t["request"]},
                ],
                "tools": t["tools"],
            }
            for t in agent
        ],
        "math": load_gsm8k(n_per_workload),
    }
    path = "/tmp/benchmark-prompts.json"
    with open(path, "w") as f:
        json.dump(prompts, f)
    return await File.from_local(path)
# {{/docs-fragment prompts}}


WARMUP = {"messages": [{"role": "user", "content": "Count from one to twenty in words."}]}


@dataclass
class RequestResult:
    output: str
    output_tokens: int
    ttft_s: float
    latency_s: float


async def _stream_one(client, base_url: str, prompt: dict, max_tokens: int) -> RequestResult:
    """Send one greedy, streaming chat completion and time it."""
    body = {
        "model": TARGET_MODEL,
        "messages": prompt["messages"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **NO_THINKING,
    }
    if prompt.get("tools"):
        body["tools"] = prompt["tools"]
    start = time.perf_counter()
    first = None
    pieces: list[str] = []
    tokens = 0
    async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    fn = call.get("function", {})
                    text += (fn.get("name") or "") + (fn.get("arguments") or "")
                if text:
                    first = first or time.perf_counter()
                    pieces.append(text)
    end = time.perf_counter()
    return RequestResult("".join(pieces), tokens, (first or end) - start, end - start)


def _summarize(results: list[RequestResult], wall_s: float) -> dict:
    tpot = [
        (r.latency_s - r.ttft_s) / (r.output_tokens - 1) for r in results if r.output_tokens > 1
    ]
    return {
        "requests": len(results),
        "output_tokens": sum(r.output_tokens for r in results),
        "throughput_tok_s": sum(r.output_tokens for r in results) / wall_s,
        "per_request_tok_s": statistics.mean(
            r.output_tokens / r.latency_s for r in results if r.latency_s > 0
        ),
        "ttft_p50_s": statistics.median(r.ttft_s for r in results),
        "tpot_p50_ms": 1000 * statistics.median(tpot) if tpot else 0.0,
    }


# {{docs-fragment benchmark}}
@bench_env.task(cache="auto", report=True)
async def benchmark_server(
    config: ServerConfig,
    prompts: File,
    concurrencies: list[int] = [1, 4, 16],
    max_tokens: int = 256,
) -> File:
    """Serve the target with one configuration and replay every workload."""
    import httpx

    drafter_path = await config.drafter.download() if config.drafter else None
    server = start_server(
        speculative_config=config.speculative_config(drafter_path),
        extra_args=[*TOOL_ARGS, "--max-model-len", "8192", "--gpu-memory-utilization", "0.9"],
    )
    with open(await prompts.download()) as f:
        workloads = json.load(f)

    report: dict = {"config": config.name, "workloads": {}}

    async def measure(client, items: list[dict], concurrency: int) -> dict:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(p):
            async with semaphore:
                return await _stream_one(client, server.base_url, p, max_tokens)

        before = read_counters(server.base_url)
        start = time.perf_counter()
        results = await asyncio.gather(*(one(p) for p in items))
        wall = time.perf_counter() - start
        spec = read_counters(server.base_url) - before
        return {
            **_summarize(results, wall),
            "acceptance_length": spec.acceptance_length,
            "acceptance_rate": spec.acceptance_rate,
            # Fraction of verification steps that accepted the token at position k.
            "acceptance_by_position": {
                str(k): v / spec.drafts for k, v in sorted(spec.accepted_per_pos.items())
            } if spec.drafts else {},
            "outputs": [r.output for r in results],
        }

    try:
        async with httpx.AsyncClient() as client:
            # Warm up CUDA graphs with a prompt outside every workload, so nothing measured
            # has been seen before.
            await asyncio.gather(*(_stream_one(client, server.base_url, WARMUP, 32) for _ in range(4)))
            for name, items in workloads.items():
                # Each concurrency level gets its own slice of prompts. A method that caches
                # earlier responses (suffix decoding) must never be measured on a prompt it
                # has already answered.
                runs = {}
                for i, c in enumerate(concurrencies):
                    runs[str(c)] = await measure(client, items[i :: len(concurrencies)], c)
                    print(f"{config.name} {name} c={c}: {runs[str(c)]['per_request_tok_s']:.1f} tok/s/request, AL={runs[str(c)]['acceptance_length']:.2f}")
                if config.method == "":
                    # The noise floor: the baseline, which caches nothing, replays its
                    # concurrency-1 prompts at the highest concurrency.
                    rerun = await measure(client, items[0 :: len(concurrencies)], concurrencies[-1])
                    runs["noise_floor_outputs"] = rerun["outputs"]
                report["workloads"][name] = runs
            if "math" in workloads:
                report["math_answers"] = [p["answer"] for p in workloads["math"][0 :: len(concurrencies)]]
    finally:
        server.stop()

    await flyte.report.replace.aio(server_report(report, config.speculative_config(drafter_path)))
    await flyte.report.flush.aio()
    path = f"/tmp/bench-{config.name}.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return await File.from_local(path)
# {{/docs-fragment benchmark}}
