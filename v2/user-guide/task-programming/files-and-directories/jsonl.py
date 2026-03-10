# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-jsonl>=2.0.0",
# ]
# main = "process_jsonl"
# params = ""
# ///

# {{docs-fragment setup}}
import flyte
from flyteplugins.jsonl import JsonlDir, JsonlFile

env = flyte.TaskEnvironment(
    name="jsonl-examples",
    image=flyte.Image.from_debian_base(name="jsonl").with_pip_packages(
        "flyteplugins-jsonl>=2.0.0"
    ),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)
# {{/docs-fragment setup}}


# {{docs-fragment write-jsonl-file}}
@env.task
@env.task
async def write_records() -> JsonlFile:
    out = JsonlFile(path="results.jsonl")

    async with out.writer() as writer:
        for i in range(500_000):
            await writer.write({"id": i, "score": i * 0.1})

    return out
# {{/docs-fragment write-jsonl-file}}


# {{docs-fragment read-jsonl-file}}
@env.task
async def read_records(data: JsonlFile) -> int:
    """Read records from a JsonlFile and return the count."""
    count = 0
    async for record in data.iter_records():
        print(record)
        count += 1
    return count
# {{/docs-fragment read-jsonl-file}}


# {{docs-fragment write-jsonl-dir}}
@env.task
async def write_large_dataset() -> JsonlDir:
    """Write a large dataset to a sharded JsonlDir.

    JsonlDir automatically rotates to a new shard file once the
    current shard reaches the size limit.
    """
    out = JsonlDir(path="dataset/")
    async with out.writer(max_bytes=1024 * 1024) as writer:  # 1 MB shards
        for i in range(1000):
            await writer.write({"index": i, "value": i * i})
    return out
# {{/docs-fragment write-jsonl-dir}}


# {{docs-fragment read-jsonl-dir}}
@env.task
async def sum_values(dataset: JsonlDir) -> int:
    """Read all records across all shards and compute a sum."""
    total = 0
    async for record in dataset.iter_records():
        total += record["value"]
    return total
# {{/docs-fragment read-jsonl-dir}}


# {{docs-fragment process-jsonl}}
@env.task
async def process_jsonl():
    data = await write_records()
    count = await read_records(data=data)
    print(f"Read {count} records")

    dataset = await write_large_dataset()
    total = await sum_values(dataset=dataset)
    print(f"Sum of values: {total}")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(process_jsonl)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment process-jsonl}}
