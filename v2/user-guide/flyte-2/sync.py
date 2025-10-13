import flyte

env = flyte.TaskEnvironment("hello_world")

@env.task
def hello_world(name: str) -> str:
    return f"Hello, {name}!"

@env.task
def main(name: str) -> str:
    for i in range(10):
        hello_world(name)
    return "Done"

if __name__ == "__main__":
    flyte.init()
    flyte.run(main, name="World")
