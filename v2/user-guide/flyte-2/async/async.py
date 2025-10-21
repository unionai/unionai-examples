# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment async}}
import asyncio
import flyte

env = flyte.TaskEnvironment("data_pipeline")


@env.task
async def process_chunk(chunk_id: int, data: str) -> str:
    # This could be any computational work - CPU or I/O bound
    await asyncio.sleep(1)  # Simulating work
    return f"Processed chunk {chunk_id}: {data}"


@env.task
async def parallel_pipeline(data_chunks: List[str]) -> List[str]:
    # Create coroutines for all chunks
    tasks = []
    for i, chunk in enumerate(data_chunks):
        tasks.append(process_chunk(i, chunk))

    # Execute all chunks in parallel
    results = await asyncio.gather(*tasks)
    return results
# {{/docs-fragment async}}


# {{docs-fragment calling-sync-from-async}}
@env.task
def legacy_computation(x: int) -> int:
    # Existing synchronous function works unchanged
    return x * x + 2 * x + 1


@env.task
async def modern_workflow(numbers: List[int]) -> List[int]:
    # Call sync tasks from async context using .aio()
    tasks = []
    for num in numbers:
        tasks.append(legacy_computation.aio(num))

    results = await asyncio.gather(*tasks)
    return results
# {{/docs-fragment calling-sync-from-async}}


# {{docs-fragment sync-map}}
@env.task
def sync_map_example(n: int) -> List[str]:
    # Synchronous version for easier migration
    results = []
    for result in flyte.map(process_item, range(n)):
        if isinstance(result, Exception):
            raise result
        results.append(result)
    return results
# {{/docs-fragment sync-map}}


# {{docs-fragment async-map}}
@env.task
async def async_map_example(n: int) -> List[str]:
    # Async version using flyte.map
    results = []
    async for result in flyte.map.aio(process_item, range(n)):
        if isinstance(result, Exception):
            raise result
        results.append(result)
    return results
# {{/docs-fragment async-map}}


@env.task
async def main():
    result = await parallel_pipeline(data_chunks=["data1", "data2", "data3"])
    print(f"parallel_pipeline result: {result}")

    result = await modern_workflow(numbers=[1, 2, 3, 4, 5])
    print(f"calling sync from async result: {result}")


    result = await async_map_example(n=5)
    print(f"async map result: {result}")

    result = sync_map_example(n=5)
    print(f"sync map result: {result}")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
