# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "World"
# ///

# {{docs-fragment all}}
# https://github.com/unionai/unionai-examples/blob/main/v2/user-guide/flyte-2/async.py

import flyte

env = flyte.TaskEnvironment("async_example_env")

@env.task
async def hello_world(name: str) -> str:
    return f"Hello, {name}!"

@env.task
async def main(name: str) -> str:
    results = []
    for i in range(10):
        results.append(hello_world(name))
    await asyncio.gather(*results)
    return "Done"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, name="World")
    print(run.name)
    print(run.url)
    run.wait()
# {{/docs-fragment all}}
