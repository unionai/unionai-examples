# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# main = "main_dspark"
# params = "num_chat=64 num_agent=64 num_shards=2 max_steps=20"
# ///
"""Train a DSpark drafter for Qwen3-8B from scratch and compare it with DFlash and EAGLE-3.

DSpark extends DFlash's block drafting with two heads. A Markov head models dependencies
between neighboring tokens in the block, which raises accuracy at later positions. A
confidence head predicts which positions the target will accept.

    build_dataset (cached) -> train_drafter(method="dspark")

`compare.py` benchmarks it against every other family, and `serve.py --method dspark` serves it.

    python dspark.py            # full run
    python dspark.py --smoke    # small smoke run
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
async def main_dspark(
    num_chat: int = 4000,
    num_agent: int = 4000,
    num_shards: int = 4,
    max_steps: int | None = None,
) -> Dir:
    """Train a DSpark drafter from scratch on the shared dataset."""
    # Same inputs as the EAGLE-3 pipeline, so every data shard is a cache hit.
    dataset = await build_dataset(num_chat, num_agent, num_shards)
    return await train_drafter(data=dataset.files, method="dspark", max_steps=max_steps)
# {{/docs-fragment main}}




if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main_dspark, **(SMOKE if "--smoke" in sys.argv else FULL))
    print(run.url)
