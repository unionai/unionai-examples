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

