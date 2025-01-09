from time import sleep
import os

import union

image = union.ImageSpec(
    registry=os.environ.get("DOCKER_REGISTRY", None),
    packages=["union"],
)

actor = union.ActorEnvironment(
    name="my-actor",
    container_image=image,
    replica_count=1,
)


class MyObj:
    def __init__(self, state: int):
        self.state = state

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state


@union.actor_cache
def get_state(obj: MyObj) -> int:
    sleep(2)
    return obj.state


@actor.task
def construct_and_get_value(state: int) -> int:
    obj = MyObj(state=state)
    return get_state(obj)


@union.workflow
def wf(state: int = 2) -> int:
    value = construct_and_get_value(state=state)
    value = construct_and_get_value(state=value)
    value = construct_and_get_value(state=value)
    value = construct_and_get_value(state=value)
    return value