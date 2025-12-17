import flyte

env = flyte.TaskEnvironment(name="simple_env")

@env.task
async def my_task(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
