import flyte


env = flyte.TaskEnvironment(
    name="hello_world",
)


@env.task
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@env.task
def to_upper(greeting: str) -> str:
    return greeting.upper()


@env.task
def main(name: str) -> str:
    greeting = say_hello(name=name)
    greeting = f"{greeting}, welcome to Flyte!"
    if len(name) < 1:
        return "No greeting for you!"
    return to_upper(greeting=greeting)
