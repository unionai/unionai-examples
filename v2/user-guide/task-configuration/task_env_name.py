env = flyte.TaskEnvironment(name="my_env")

@env.task
async def my_task(data: str) -> str:
