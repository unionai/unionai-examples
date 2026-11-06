# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# main = "main_dflash"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20 n_bench=12"
# ///
"""Fine-tune a DFlash block drafter for Qwen3-8B and compare it with EAGLE-3.

DFlash drafts a block of 16 positions in one forward pass of a small block-diffusion model,
instead of one token per pass like EAGLE-3. When the target accepts a long run of drafted
tokens, a block drafter pays for one draft pass instead of several.

    build_dataset (cached) -> train_drafter(method="dflash") -> benchmark -> report

    python dflash.py            # full run
    python dflash.py --smoke    # small smoke run
    python dflash.py --serve    # serve the trained drafter
"""

import sys

import flyte
import flyte.app
from flyte.io import Dir

from benchmark import ServerConfig, drafter_config, prepare_benchmark_prompts
from config import driver_env
from data import build_dataset
from pipeline import SMOKE, run_benchmarks
from recipes import DFLASH
from serve import llm_app
from train import train_drafter


# {{docs-fragment main}}
@driver_env.task
async def main_dflash(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    epochs: int = 2,
    max_steps: int | None = None,
    n_bench: int = 78,
) -> Dir:
    """Fine-tune the public DFlash head on the shared dataset and benchmark it."""
    # Same inputs as the EAGLE-3 pipeline, so every data shard is a cache hit.
    dataset = await build_dataset(num_chat, num_agent, num_shards)
    drafter = await train_drafter(
        data=dataset.files, method="dflash", epochs=epochs, max_steps=max_steps
    )
    prompts = await prepare_benchmark_prompts(n_bench)
    await run_benchmarks(
        [
            ServerConfig(name="baseline"),
            drafter_config("eagle3-public", "eagle3"),
            drafter_config("dflash-public", "dflash"),
            drafter_config("dflash-finetuned", "dflash", drafter),
        ],
        prompts,
    )
    return drafter
# {{/docs-fragment main}}


# {{docs-fragment app}}
dflash_app = llm_app.clone_with(
    name="qwen3-8b-dflash",
    draft_model_path=flyte.app.RunOutput(type="directory", task_name="specdec-driver.main_dflash"),
    speculative_config={"method": "dflash", "num_speculative_tokens": DFLASH.num_speculative_tokens},
)
# {{/docs-fragment app}}


if __name__ == "__main__":
    flyte.init_from_config()
    if "--serve" in sys.argv:
        print(flyte.with_servecontext(activate_timeout=1800.0).serve(dflash_app).url)
    else:
        run = flyte.run(main_dflash, **(SMOKE if "--smoke" in sys.argv else {}))
        print(run.url)
