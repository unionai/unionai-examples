# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "World"
# ///

# {{docs-fragment all}}
# https://github.com/unionai/unionai-examples/blob/main/v2/user-guide/flyte-2/sync.py

import flyte

env = flyte.TaskEnvironment("sync_example_env")

@env.task
def hello_world(name: str) -> str:
    return f"Hello, {name}!"

@env.task
def main(name: str) -> str:
    for i in range(10):
        hello_world(name)
    return "Done"

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, name="World")
    print(run.name)
    print(run.url)
    run.wait()
# {{docs-fragment all}}
