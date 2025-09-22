# {{docs-fragment import-and-env}}
import flyte
import random
from datetime import timedelta

env = flyte.TaskEnvironment(name="my-env")
# {{/docs-fragment import-and-env}}


# {{docs-fragment retry}}
@env.task(retries=3)
async def unreliable() -> str:
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Task failed!")
    return "Success!"


@env.task
async def driver() -> str:
    return await unreliable()
# {{/docs-fragment retry}}


# {{docs-fragment retry-override}}
@env.task
async def driver_override() -> str:
    return await unreliable.override(retries=5)()
# {{/docs-fragment retry-override}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(driver)
    print(run.name)
    print(run.url)
    run.wait(run)

    run_2 = flyte.run(driver_override)
    print(run_2.name)
    print(run_2.url)
    run_2.wait(run_2)
# {{docs-fragment run}}
