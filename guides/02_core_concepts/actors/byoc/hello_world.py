import os

from flytekit import workflow, Resources, ImageSpec
from union.actor import ActorEnvironment

image = ImageSpec(
    registry=os.environ.get("DOCKER_REGISTRY", None),
    packages=["union"],
)

actor = ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=30,
    requests=Resources(
        cpu="2",
        mem="300Mi",
    ),
    container_image=image,
)


@actor.task
def say_hello() -> str:
    return "hello"


@workflow
def wf():
    say_hello()
