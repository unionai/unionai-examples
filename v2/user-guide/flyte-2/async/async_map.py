@env.task
async def async_map_example(n: int) -> List[str]:
    # Async version using flyte.map
    results = []
    async for result in flyte.map.aio(process_item, range(n)):
        if isinstance(result, Exception):
            raise result
        results.append(result)
    return results
