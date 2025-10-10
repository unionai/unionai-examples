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