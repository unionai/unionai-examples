# {{docs-fragment basic}}
import asyncio

import flyte

env = flyte.TaskEnvironment("large_fanout")


@env.task
async def my_task(x: int) -> int:
    return x


@env.task
async def main(r: int):
    results = []
    for i in range(r):
        results.append(my_task(x=i))
    result = await asyncio.gather(*results)

    return result


if __name__ == "__main__":
    flyte.init_from_config("config.yaml")
    run = flyte.run(main, r=50)
    print(run.url)
    run.wait(run)
# {{/docs-fragment basic}}

# {{docs-fragment parallel}}
@env.task
async def parallel_fanout_example(n: int) -> List[str]:
    results = []

    # Collect all task invocations first
    for i in range(n):
        results.append(my_async_task(i))

    # Execute all tasks in parallel
    final_results = await asyncio.gather(*results)

    return final_results
# {{/docs-fragment parallel}}

# {{docs-fragment sequential}}
@env.task
async def sequential_fanout_example(n: int) -> List[str]:
    results = []

    # Execute tasks one at a time in sequence
    for i in range(n):
        result = await my_async_task(i)  # Await each task individually
        results.append(result)

    return results
# {{/docs-fragment sequential}}

# {{docs-fragment mixed}}
@env.task
async def mixed_fanout_example(n: int) -> Tuple[List[str], List[str]]:
    # First: parallel execution
    parallel_tasks = []
    for i in range(n):
        parallel_tasks.append(fast_task(i))
    parallel_results = await asyncio.gather(*parallel_tasks)

    # Second: sequential execution using results from parallel phase
    sequential_results = []
    for result in parallel_results:
        processed = await slow_processing_task(result)
        sequential_results.append(processed)

    return parallel_results, sequential_results
# {{/docs-fragment mixed}}

# {{docs-fragment multi-phase}}
@env.task
async def multi_phase_workflow(data_size: int) -> List[int]:
    # First phase: data preprocessing
    preprocessed = []
    for i in range(data_size):
        preprocessed.append(preprocess_task(i))
    phase1_results = await asyncio.gather(*preprocessed)

    # Second phase: main processing
    processed = []
    for result in phase1_results:
        processed.append(process_task(result))
    phase2_results = await asyncio.gather(*processed)

    # Third phase: postprocessing
    postprocessed = []
    for result in phase2_results:
        postprocessed.append(postprocess_task(result))
    final_results = await asyncio.gather(*postprocessed)

    return final_results
# {{/docs-fragment multi-phase}}

# {{docs-fragment batching}}
# For very large fanouts, consider batching
batch_size = 100
for i in range(0, total_items, batch_size):
    batch = items[i:i + batch_size]
    batch_results = []
    for item in batch:
        batch_results.append(process_task(item))
    await asyncio.gather(*batch_results)
# {{/docs-fragment batching}}

# {{docs-fragment errors}}

# Use return_exceptions=True to handle failures gracefully
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Task {i} failed: {result}")
# {{/docs-fragment errors}}

# {{docs-fragment memory}}
# Process in chunks to manage memory
chunk_size = 1000
all_results = []
for chunk_start in range(0, total_size, chunk_size):
    chunk_tasks = []
    for i in range(chunk_start, min(chunk_start + chunk_size, total_size)):
        chunk_tasks.append(my_task(i))
    chunk_results = await asyncio.gather(*chunk_tasks)
    all_results.extend(chunk_results)
# {{/docs-fragment memory}}
