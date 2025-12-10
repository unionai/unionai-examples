# unionai-examples/v2/user-guide/task-deployment/deployment-patterns/dynamic_environments/environment_picker.py

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
