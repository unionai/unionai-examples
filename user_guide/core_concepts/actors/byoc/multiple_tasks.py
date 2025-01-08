import os

import union

image = union.ImageSpec(
    registry=os.environ.get("DOCKER_REGISTRY", None),
    packages=["union"],
)

actor = union.ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=30,
    requests=union.Resources(cpu="1", mem="450Mi"),
    container_image=image,
)


@actor.task
def say_hello(name: str) -> str:
    return f"hello {name}"


@actor.task
def scream_hello(name: str) -> str:
    return f"HELLO {name}"


@union.workflow
def my_child_wf(name: str) -> str:
    return scream_hello(name=name)


my_child_wf_lp = union.LaunchPlan.get_default_launch_plan(union.current_context(), my_child_wf)


@union.workflow
def my_parent_wf(name: str) -> str:
    a = say_hello(name=name)
    b = my_child_wf(name=a)
    return my_child_wf_lp(name=b)
