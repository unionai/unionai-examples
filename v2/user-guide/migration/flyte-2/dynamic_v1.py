from flytekit import task, workflow, dynamic


@task
def list_partitions(n: int) -> list[int]:
    return list(range(n))


@task
def process_partition(partition_id: int) -> int:
    # Aggregate one data partition.
    return partition_id * 2


@dynamic
def process_all(partitions: list[int]) -> list[int]:
    results = []
    for partition_id in partitions:
        results.append(process_partition(partition_id=partition_id))
    return results


@workflow
def main(n: int) -> list[int]:
    partitions = list_partitions(n=n)
    return process_all(partitions=partitions)
