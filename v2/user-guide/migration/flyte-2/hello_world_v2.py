# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "name=World"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@env.task
def to_upper(greeting: str) -> str:
    return greeting.upper()


# The "workflow" is now just a task that calls other tasks.
@env.task
def main(name: str) -> str:
    greeting = say_hello(name)
    return to_upper(greeting)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, name="World")
    print(r.name)
    print(r.url)
    r.wait()
