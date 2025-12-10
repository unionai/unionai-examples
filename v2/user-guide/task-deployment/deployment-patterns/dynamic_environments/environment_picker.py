"""
An example of how to pick different environments based on domain.
NOTE: You cannot run this example here directly, because flyte.init() needs to be called first.

flyte.init() invocation at the module level is strictly discouraged. The reason is runtime, flyte controls the
 initialization and configuration file is not present at runtime.

To run an example like this programmatically see main.py. Otherwise flyte run and flyte deploy should work.
"""

import os
import flyte

def create_env():
    """Create environment based on deployment domain"""
    if flyte.current_domain() == "development":
        return flyte.TaskEnvironment(
            name="dev",
            image=flyte.Image.from_debian_base(),
            env_vars={"MY_ENV": "dev"}
        )
    return flyte.TaskEnvironment(
        name="prod",
        image=flyte.Image.from_production_base(),
        env_vars={"MY_ENV": "prod"}
    )

env = create_env()

@env.task
async def my_task(n: int) -> int:
    print(f"Environment: {os.environ['MY_ENV']}")
    return n + 1

@env.task
async def entrypoint(n: int) -> int:
    return await my_task(n)
