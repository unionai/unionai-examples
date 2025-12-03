# {{docs-fragment simple-file}}
# my_example.py
import flyte

env = flyte.TaskEnvironment(name="simple_env")

@env.task
async def my_task(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(my_task, name="World")
    print(run.url)
# {{/docs-fragment simple-file}}