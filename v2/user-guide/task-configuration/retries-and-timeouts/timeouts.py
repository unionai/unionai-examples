# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment import-and-env}}
import random
from datetime import timedelta
import asyncio

import flyte
from flyte import Timeout

env = flyte.TaskEnvironment(name="my-env")
# {{/docs-fragment import-and-env}}


# {{docs-fragment timeout-seconds}}
@env.task(timeout=60)  # 60 seconds
async def timeout_seconds() -> str:
    await asyncio.sleep(random.randint(0, 120))  # Random wait between 0 and 120 seconds
    return "timeout_seconds completed"
# {{/docs-fragment timeout-seconds}}


# {{docs-fragment timeout-timedelta}}
@env.task(timeout=timedelta(minutes=1))
async def timeout_timedelta() -> str:
    await asyncio.sleep(random.randint(0, 120))  # Random wait between 0 and 120 seconds
    return "timeout_timedelta completed"
# {{/docs-fragment timeout-timedelta}}


# {{docs-fragment timeout-advanced}}
@env.task(timeout=Timeout(
    max_runtime=timedelta(minutes=1),      # Max execution time per attempt
    max_queued_time=timedelta(minutes=1)   # Max time in queue before starting
))
async def timeout_advanced() -> str:
    await asyncio.sleep(random.randint(0, 120))  # Random wait between 0 and 120 seconds
    return "timeout_advanced completed"
# {{/docs-fragment timeout-advanced}}


# {{docs-fragment timeout-with-retry}}
@env.task(
    retries=3,
    timeout=Timeout(
        max_runtime=timedelta(minutes=1),
        max_queued_time=timedelta(minutes=1)
    )
)
async def timeout_with_retry() -> str:
    await asyncio.sleep(random.randint(0, 120))  # Random wait between 0 and 120 seconds
    return "timeout_advanced completed"
# {{/docs-fragment timeout-with-retry}}

# {{docs-fragment main}}
@env.task
async def main() -> list[str]:
    tasks = [
        timeout_seconds(),
        timeout_seconds.override(timeout=120)(),  # Override to 120 seconds
        timeout_timedelta(),
        timeout_advanced(),
        timeout_with_retry(),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for r in results:
        if isinstance(r, Exception):
            output.append(f"Failed: {r}")
        else:
            output.append(r)
    return output
# {{/docs-fragment main}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}