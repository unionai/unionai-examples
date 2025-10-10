env = flyte.TaskEnvironment(
    name="my_env",
    resources=Resources(cpu=1),
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=300)
)

@env.task
async def my_task(data: str) -> str:
    ...

@env.task
async def main_workflow() -> str:
    # `my_task.override(resources=Resources(cpu=4))` will fail. Instead use:
    result = await my_task.override(reusable="off", resources=Resources(cpu=4))
