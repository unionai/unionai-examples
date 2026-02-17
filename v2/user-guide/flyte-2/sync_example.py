# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "name='World'"
# ///

# {{docs-fragment all}}
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
    r = flyte.run(main, name="World")
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}
