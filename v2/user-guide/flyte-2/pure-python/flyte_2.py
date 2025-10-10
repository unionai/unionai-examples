import flytekit

image = flytekit.ImageSpec(
    name="hello-world-image",
    packages=[...],
)

@flytekit.task(container_image=image)
def mean(data: list[float]) -> float:
    return sum(list) / len(list)

@flytekit.workflow
def main(data: list[float]) -> float:
    output = mean(data)

    # ❌ performing trivial operations in a workflow is not allowed
    # output = output / 100

    # ❌ if/else is not allowed
    # if output < 0:
    #     raise ValueError("Output cannot be negative")

    return output