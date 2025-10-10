import flyte

env = flyte.TaskEnvironment("hello_world")

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
    flyte.init()
    flyte.run(main, name="World")
