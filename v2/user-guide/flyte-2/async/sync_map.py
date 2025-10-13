@env.task
def sync_map_example(n: int) -> List[str]:
    # Synchronous version for easier migration
    results = []
    for result in flyte.map(process_item, range(n)):
        if isinstance(result, Exception):
            raise result
        results.append(result)
    return results
