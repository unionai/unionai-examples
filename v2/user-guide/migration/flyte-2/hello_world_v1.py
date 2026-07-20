import flytekit


@flytekit.task
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@flytekit.task
def to_upper(greeting: str) -> str:
    return greeting.upper()


@flytekit.workflow
def main(name: str) -> str:
    greeting = say_hello(name=name)
    return to_upper(greeting=greeting)
