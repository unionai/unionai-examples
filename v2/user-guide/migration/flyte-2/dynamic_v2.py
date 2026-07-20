# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "n=5"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="dynamic")


@env.task
def process_partition(partition_id: int) -> int:
    # Aggregate one data partition.
    return partition_id * 2


# No @dynamic decorator needed: a plain task can loop over runtime data (e.g. a
# variable number of partitions discovered at runtime) and call other tasks.
@env.task
def main(n: int) -> list[int]:
    return [process_partition(partition_id) for partition_id in range(n)]
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, n=5)
    print(r.name)
    print(r.url)
    r.wait()
