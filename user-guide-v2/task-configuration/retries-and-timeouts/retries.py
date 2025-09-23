# {{docs-fragment import-and-env}}
import flyte
import random
from datetime import timedelta

env = flyte.TaskEnvironment(name="my-env")
# {{/docs-fragment import-and-env}}


# {{docs-fragment retry}}
@env.task(retries=3)
async def retry() -> str:
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Task failed!")
    return "Success!"


@env.task
async def main() -> str:
    s_1 = await retry()
    s_2 = await retry.override(retries=5)()
    return s_1 + " & " + s_2
# {{/docs-fragment retry}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    run.wait(run)
# {{docs-fragment run}}
