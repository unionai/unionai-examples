# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# main = "main"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20"
# ///
"""Generate target-generated training data and fine-tune an EAGLE-3 draft head for Qwen3-8B.

    build_dataset -> train_drafter(method="eagle3")

`compare.py` benchmarks the result against DFlash, DSpark, and draft-free methods.

    python pipeline.py            # full run
    python pipeline.py --smoke    # small smoke run
"""

import sys

import flyte
from flyte.io import Dir

from config import driver_env
from data import build_dataset
from train import train_drafter

# Every pipeline trains on the same dataset. Identical inputs make every data shard, and
# every drafter already trained with the same arguments, a cache hit.
FULL = dict(num_chat=4000, num_agent=4000, num_shards=4, max_steps=None)
SMOKE = dict(num_chat=64, num_agent=64, num_shards=2, max_steps=20)


# {{docs-fragment main}}
@driver_env.task
async def main(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    max_steps: int | None = None,
) -> Dir:
    """Generate the dataset and fine-tune the public EAGLE-3 head on it."""
    dataset = await build_dataset(num_chat, num_agent, num_shards)
    return await train_drafter(data=dataset.files, method="eagle3", max_steps=max_steps)
# {{/docs-fragment main}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, **(SMOKE if "--smoke" in sys.argv else FULL))
    print(run.url)
# {{/docs-fragment run}}
