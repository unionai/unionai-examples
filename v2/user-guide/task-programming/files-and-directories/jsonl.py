# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyteplugins-jsonl>=2.0.4",
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
        "flyteplugins-jsonl"
    ),
)
# {{/docs-fragment setup}}


# {{docs-fragment write-jsonl-file}}
@env.task
async def write_records() -> JsonlFile:
    """Write records to a single JSONL file."""
    out = JsonlFile.new_remote("results.jsonl")
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
        count += 1
    return count
# {{/docs-fragment read-jsonl-file}}


# {{docs-fragment write-compressed-file}}
@env.task
async def write_compressed() -> JsonlFile:
    """Write a zstd-compressed JSONL file.

    Compression is activated by using a .jsonl.zst extension.
    Both reading and writing handle compression transparently.
    """
    out = JsonlFile.new_remote("results.jsonl.zst")
    async with out.writer(compression_level=3) as writer:
        for i in range(100_000):
            await writer.write({"id": i, "compressed": True})
    return out


# {{/docs-fragment write-compressed-file}}


# {{docs-fragment error-handling}}
@env.task
async def read_with_error_handling(data: JsonlFile) -> int:
    """Read records, skipping any corrupt lines instead of raising."""
    count = 0
    async for record in data.iter_records(on_error="skip"):
        count += 1
    return count


@env.task
async def read_with_custom_handler(data: JsonlFile) -> int:
    """Read records with a custom error handler that collects errors."""
    errors: list[dict] = []

    def on_error(line_number: int, raw_line: bytes, exc: Exception) -> None:
        errors.append({"line": line_number, "error": str(exc)})

    count = 0
    async for record in data.iter_records(on_error=on_error):
        count += 1
    print(f"{count} valid records, {len(errors)} errors")
    return count


# {{/docs-fragment error-handling}}


# {{docs-fragment write-jsonl-dir}}
@env.task
async def write_large_dataset() -> JsonlDir:
    """Write a large dataset to a sharded JsonlDir.

    JsonlDir automatically rotates to a new shard file once the
    current shard reaches the record or byte limit. Shards are named
    part-00000.jsonl, part-00001.jsonl, etc.
    """
    out = JsonlDir.new_remote("dataset/")
    async with out.writer(
        max_records_per_shard=100_000,
        max_bytes_per_shard=256 * 1024 * 1024,  # 256 MB
    ) as writer:
        for i in range(500_000):
            await writer.write({"index": i, "value": i * i})
    return out
# {{/docs-fragment write-jsonl-dir}}


# {{docs-fragment write-compressed-dir}}
@env.task
async def write_compressed_dir() -> JsonlDir:
    """Write zstd-compressed shards by specifying the shard extension."""
    out = JsonlDir.new_remote("compressed_dataset/")
    async with out.writer(
        shard_extension=".jsonl.zst",
        max_records_per_shard=50_000,
    ) as writer:
        for i in range(200_000):
            await writer.write({"id": i, "data": f"payload-{i}"})
    return out


# {{/docs-fragment write-compressed-dir}}


# {{docs-fragment read-jsonl-dir}}
@env.task
async def sum_values(dataset: JsonlDir) -> int:
    """Read all records across all shards and compute a sum.

    Iteration is transparent across shards and handles mixed
    compressed/uncompressed shards automatically. The next shard is
    prefetched in the background for higher throughput.
    """
    total = 0
    async for record in dataset.iter_records():
        total += record["value"]
    return total
# {{/docs-fragment read-jsonl-dir}}


# {{docs-fragment batch-iteration}}
@env.task
async def process_in_batches(dataset: JsonlDir) -> int:
    """Process records in batches of dicts for bulk operations."""
    total = 0
    async for batch in dataset.iter_batches(batch_size=1000):
        # Each batch is a list[dict]
        total += len(batch)
    return total


# {{/docs-fragment batch-iteration}}


# {{docs-fragment arrow-batches}}
arrow_env = flyte.TaskEnvironment(
    name="jsonl-arrow",
    image=flyte.Image.from_debian_base(name="jsonl-arrow").with_pip_packages(
        "flyteplugins-jsonl[arrow]"
    ),
)


@arrow_env.task
async def analyze_with_arrow(dataset: JsonlDir) -> float:
    """Stream records as Arrow RecordBatches for analytics.

    Memory usage is bounded by batch_size — the full dataset is
    never loaded into memory at once.
    """
    import pyarrow as pa

    batches = []
    async for batch in dataset.iter_arrow_batches(batch_size=65_536):
        batches.append(batch)

    table = pa.Table.from_batches(batches)
    mean_value = table.column("value").to_pylist()
    return sum(mean_value) / len(mean_value)


# {{/docs-fragment arrow-batches}}


# {{docs-fragment process-jsonl}}
@env.task
async def process_jsonl():
    # Write and read a single JSONL file
    data = await write_records()
    count = await read_records(data=data)
    print(f"Read {count} records")

    # Write and read a sharded directory
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
