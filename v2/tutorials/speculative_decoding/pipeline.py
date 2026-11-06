# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "httpx>=0.27",
# ]
# main = "main"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20 n_bench=12"
# ///
"""Train an EAGLE-3 draft head for Qwen3-8B and prove it is faster and lossless.

    build_dataset -> train_drafter -> benchmark every configuration -> report

Run the whole pipeline with `python pipeline.py`, or a small smoke run with
`python pipeline.py --smoke`.
"""

import asyncio
import json
import sys

import flyte
from flyte.io import Dir, File

from benchmark import ServerConfig, benchmark_server, default_configs, prepare_benchmark_prompts
from config import driver_env
from data import build_dataset
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

    await flyte.report.replace.aio(render_html(summary))
    await flyte.report.flush.aio()

    path = "/tmp/summary.json"
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return await File.from_local(path)
# {{/docs-fragment run-benchmarks}}


# {{docs-fragment main}}
@driver_env.task
async def main(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    epochs: int = 2,
    max_steps: int | None = None,
    n_bench: int = 78,
) -> Dir:
    """Generate data, fine-tune the public EAGLE-3 head on it, and benchmark the result."""
    dataset = await build_dataset(num_chat, num_agent, num_shards)

    drafter = await train_drafter(data=dataset.files, epochs=epochs, max_steps=max_steps)

    prompts = await prepare_benchmark_prompts(n_bench)
    await run_benchmarks(default_configs(trained=drafter), prompts)
    return drafter
# {{/docs-fragment main}}


SMOKE = dict(num_chat=64, num_agent=64, num_shards=2, max_steps=20, n_bench=12)

# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    kwargs = SMOKE if "--smoke" in sys.argv else {}
    run = flyte.run(main, **kwargs)
    print(run.url)
# {{/docs-fragment run}}
