import os

import union

image = union.ImageSpec(
    registry=os.environ.get("DOCKER_REGISTRY", None),
    packages=["union"],
)

actor = union.ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=300,
    requests=union.Resources(cpu="2", mem="500Mi"),
    container_image=image,
)


@actor.task
def plus_one(input: int) -> int:
    return input + 1


@union.workflow
def wf(input: int = 0) -> int:
    a = plus_one(input=input)
    b = plus_one(input=a)
    c = plus_one(input=b)
    return plus_one(input=c)
