# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# main = "main_dflash"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20"
# ///
"""Fine-tune a DFlash block drafter for Qwen3-8B and compare it with EAGLE-3.

DFlash drafts a block of 16 positions in one forward pass of a small block-diffusion model,
instead of one token per pass like EAGLE-3. When the target accepts a long run of drafted
tokens, a block drafter pays for one draft pass instead of several.

    build_dataset (cached) -> train_drafter(method="dflash")

`compare.py` benchmarks it against every other family, and `serve.py --method dflash` serves it.

    python dflash.py            # full run
    python dflash.py --smoke    # small smoke run
"""

import sys

import flyte
from flyte.io import Dir

from config import driver_env
from data import build_dataset
from pipeline import FULL, SMOKE
from train import train_drafter


# {{docs-fragment main}}
@driver_env.task
async def main_dflash(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    max_steps: int | None = None,
) -> Dir:
    """Fine-tune the public DFlash head on the shared dataset."""
    # Same inputs as the EAGLE-3 pipeline, so every data shard is a cache hit.
    dataset = await build_dataset(num_chat, num_agent, num_shards)
    return await train_drafter(data=dataset.files, method="dflash", max_steps=max_steps)
# {{/docs-fragment main}}




if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main_dflash, **(SMOKE if "--smoke" in sys.argv else FULL))
    print(run.url)
