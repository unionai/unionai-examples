# NOTE: flyte.init() invocation at the module level is strictly discouraged.
# At runtime, Flyte controls initialization and configuration files are not present.

import os

import flyte


def create_env():
    if flyte.current_domain() == "development":
        return flyte.TaskEnvironment(name="dev", image=flyte.Image.from_debian_base(), env_vars={"MY_ENV": "dev"})
    return flyte.TaskEnvironment(name="prod", image=flyte.Image.from_debian_base(), env_vars={"MY_ENV": "prod"})


env = create_env()


@env.task
async def my_task(n: int) -> int:
    print(f"Environment Variable MY_ENV = {os.environ['MY_ENV']}", flush=True)
    return n + 1


@env.task
async def entrypoint(n: int) -> int:
    print(f"Environment Variable MY_ENV = {os.environ['MY_ENV']}", flush=True)
    return await my_task(n)
