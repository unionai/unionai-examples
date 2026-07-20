from functools import partial

from flytekit import task, workflow, map_task


@task
def get_shards(n: int) -> list[int]:
    return list(range(n))


@task
def score_shard(shard_id: int, model_version: int) -> int:
    # Score one shard of records with the given model version.
    return shard_id * model_version


@workflow
def main(n: int, model_version: int) -> list[int]:
    shards = get_shards(n=n)
    return map_task(
        partial(score_shard, model_version=model_version),
        concurrency=10,
    )(shard_id=shards)
