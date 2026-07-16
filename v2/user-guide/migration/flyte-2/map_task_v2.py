# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "n=5 model_version=3"
# ///
import asyncio
from functools import partial

import flyte

env = flyte.TaskEnvironment(name="batch_scoring")


# {{docs-fragment sync}}
@env.task
def score_shard(shard_id: int, model_version: int) -> int:
    # Score one shard of records with the given model version.
    return shard_id * model_version


@env.task
def main(n: int, model_version: int) -> list[int]:
    bound = partial(score_shard, model_version=model_version)
    # flyte.map is a drop-in for map_task, but it returns a generator, so wrap
    # it in list() to materialize the results.
    return list(flyte.map(bound, range(n), concurrency=10))
# {{/docs-fragment sync}}


# {{docs-fragment async}}
@env.task
async def score_shard_async(shard_id: int, model_version: int) -> int:
    return shard_id * model_version


@env.task
async def main_async(n: int, model_version: int) -> list[int]:
    # asyncio.gather is the idiomatic Flyte 2 way to fan out.
    coros = [score_shard_async(i, model_version) for i in range(n)]
    return list(await asyncio.gather(*coros))
# {{/docs-fragment async}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, n=5, model_version=3)
    print(r.name)
    print(r.url)
    r.wait()
