import flyte

env = flyte.TaskEnvironment(
    name="my_module",
)


@env.task
async def say_hello(name: str) -> str:
    return f"Hello, {name}!"
