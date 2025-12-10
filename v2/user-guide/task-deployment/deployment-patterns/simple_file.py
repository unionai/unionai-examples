# unionai-examples/v2/user-guide/task-deployment/deployment-patterns/simple_file.py

import flyte

env = flyte.TaskEnvironment(name="simple_env")

@env.task
async def my_task(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(my_task, name="World")
    print(run.url)
