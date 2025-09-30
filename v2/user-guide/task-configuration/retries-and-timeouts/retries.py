# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# ///

# {{docs-fragment import-and-env}}
import random
from datetime import timedelta

import flyte


env = flyte.TaskEnvironment(name="my-env")
# {{/docs-fragment import-and-env}}


# {{docs-fragment retry}}
@env.task(retries=3)
async def retry() -> str:
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Task failed!")
    return "Success!"


@env.task
async def main() -> list[str]:
    results = []
    try:
        results.append(await retry())
    except Exception as e:
        results.append(f"Failed: {e}")
    try:
        results.append(await retry.override(retries=5)())
    except Exception as e:
        results.append(f"Failed: {e}")
    return results
# {{/docs-fragment retry}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    run.wait()
# {{/docs-fragment run}}
