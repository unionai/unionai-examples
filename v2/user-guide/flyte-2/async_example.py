# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "main"
# params = "name='World'"
# ///

# {{docs-fragment all}}
import asyncio
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
    r = flyte.run(main, name="World")
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}
