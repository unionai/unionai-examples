import flyte

env = flyte.TaskEnvironment(
    "hello_world",
    image=flyte.Image.from_debian_base().with_pip_packages(...),
)

@env.task
def mean(data: list[float]) -> float:
    return sum(list) / len(list)

@env.task
def main(data: list[float]) -> float:
    output = mean(data)

    # ✅ performing trivial operations in a workflow is allowed
    output = output / 100

    # ✅ if/else is allowed
    if output < 0:
        raise ValueError("Output cannot be negative")

    return output
