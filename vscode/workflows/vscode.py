"""Union workflow example of interactive tasks (@vscode)"""

import typing
from flytekit import task, workflow, ImageSpec
from flytekitplugins.flyin import vscode

image = ImageSpec(
    registry="ghcr.io/unionai-oss",
    name="interactive-tasks-example",
    base_image="ghcr.io/flyteorg/flytekit:py3.11-latest",
    requirements="requirements.txt"
)


@task(container_image=image)
@vscode
def say_hello(name: str) -> str:
    s = f"Hello, {name}!"
    return s


@workflow
def wf(name: str = "world") -> str:
    greeting = say_hello(name=name)
    return greeting