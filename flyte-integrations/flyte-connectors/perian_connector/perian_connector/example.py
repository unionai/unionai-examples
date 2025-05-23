# # Perian connector example usage
#
# {{run-on-union}}
#
# This example shows how to use the Perian connector to execute tasks on Perian Job Platform.

from flytekit import ImageSpec, task, workflow
from flytekitplugins.perian_job import PerianConfig

image_spec = ImageSpec(
    name="flyte-test",
    registry="my-registry",
    python_version="3.11",
    apt_packages=["wget", "curl", "git"],
    packages=[
        "flytekitplugins-perian-job",
    ],
)


# `PerianConfig` configures `PerianTask`. Tasks specified with `PerianConfig` will be executed on Perian Job Platform.


@task(
    container_image=image_spec,
    task_config=PerianConfig(
        accelerators=1,
        accelerator_type="A100",
    ),
)
def perian_hello(name: str) -> str:
    return f"hello {name}!"


@workflow
def my_wf(name: str = "world") -> str:
    return perian_hello(name=name)
