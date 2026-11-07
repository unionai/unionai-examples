# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# main = "compare"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20 n_bench=12"
# ///
"""Benchmark every draft model family against no speculation, on identical GPUs.

    build_dataset (cached)
      -> train_drafter for EAGLE-3, DFlash, and DSpark, in parallel (cached)
      -> benchmark eight serving configurations, one GPU each
      -> report

Training is cached, so drafters already trained by `pipeline.py`, `dflash.py`, or
`dspark.py` with the same data are reused, not retrained.

    python compare.py            # full run
    python compare.py --smoke    # small smoke run
"""

import asyncio
import json
import sys

import flyte
from flyte.io import File

from benchmark import ServerConfig, benchmark_server, drafter_config, prepare_benchmark_prompts
from config import driver_env
from data import build_dataset
from pipeline import FULL, SMOKE
from report import render_html, summarize
from train import train_drafter


# {{docs-fragment run-benchmarks}}
@driver_env.task(report=True)
async def run_benchmarks(configs: list[ServerConfig], prompts: File) -> File:
    """Benchmark every configuration in parallel, then compare them in a report."""
    with flyte.group("benchmarks"):
        files = await asyncio.gather(*(benchmark_server(c, prompts) for c in configs))

    results = []
    for f in files:
        with open(await f.download()) as fh:
            results.append(json.load(fh))
    summary = summarize(results)

    await flyte.report.replace.aio(render_html(summary, title="Speculative decoding: every draft model family"))
    await flyte.report.flush.aio()

    path = "/tmp/summary.json"
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return await File.from_local(path)
# {{/docs-fragment run-benchmarks}}


# {{docs-fragment compare}}
@driver_env.task
async def compare(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    max_steps: int | None = None,
    n_bench: int = 78,
) -> File:
    """Train (or reuse) a drafter of every family, then benchmark them all."""
    dataset = await build_dataset(num_chat, num_agent, num_shards)
    with flyte.group("train"):
        eagle3, dflash, dspark = await asyncio.gather(
            *(
                train_drafter(data=dataset.files, method=m, max_steps=max_steps)
                for m in ("eagle3", "dflash", "dspark")
            )
        )
    prompts = await prepare_benchmark_prompts(n_bench)
    configs = [
        # No speculation, and two methods that need no training.
        ServerConfig(name="baseline"),
        ServerConfig(name="ngram", method="ngram", num_speculative_tokens=4),
        ServerConfig(name="suffix", method="suffix", num_speculative_tokens=16),
        # Each family's public head, and the one trained on this dataset.
        drafter_config("eagle3-public", "eagle3"),
        drafter_config("eagle3-finetuned", "eagle3", eagle3),
        drafter_config("dflash-public", "dflash"),
        drafter_config("dflash-finetuned", "dflash", dflash),
        drafter_config("dspark-trained", "dspark", dspark),
    ]
    return await run_benchmarks(configs, prompts)
# {{/docs-fragment compare}}


if __name__ == "__main__":
    flyte.init_from_config()
    smoke = "--smoke" in sys.argv
    run = flyte.run(compare, **(SMOKE if smoke else FULL), n_bench=12 if smoke else 78)
    print(run.url)
