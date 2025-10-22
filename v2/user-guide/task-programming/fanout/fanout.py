# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
# ]
# main = "parallel_data_fetching"
# params = "user_ids = [1, 2, 3, 4, 5]"
# ///

# {{docs-fragment setup}}
import asyncio
from typing import List, Tuple

import flyte

env = flyte.TaskEnvironment("fanout_env")


@env.task
async def fetch_data(user_id: int) -> dict:
    """Simulate fetching user data from an API - good for parallel execution."""
    # Simulate network I/O delay
    await asyncio.sleep(0.1)
    return {
        "user_id": user_id,
        "name": f"User_{user_id}",
        "score": user_id * 10,
        "data": f"fetched_data_{user_id}"
    }
# {{/docs-fragment setup}} }}


# {{docs-fragment parallel}}
@env.task
async def parallel_data_fetching(user_ids: List[int]) -> List[dict]:
    """Fetch data for multiple users in parallel - ideal for I/O bound operations."""
    tasks = []

    # Collect all fetch tasks - these can run in parallel since they're independent
    for user_id in user_ids:
        tasks.append(fetch_data(user_id))

    # Execute all fetch operations in parallel
    results = await asyncio.gather(*tasks)
    return results
# {{/docs-fragment parallel}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    user_ids = [1, 2, 3, 4, 5]
    r = flyte.run(parallel_data_fetching, user_ids)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}
