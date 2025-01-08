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


@actor.cache
def load_model(state: int) -> callable:
    sleep(4)  # simulate model loading
    return lambda value: state + value


@actor.task
def evaluate(value: int, state: int) -> int:
    model = load_model(state=state)
    return model(value)


@union.workflow
def wf(init_value: int = 1, state: int = 3) -> int:
    out = evaluate(value=init_value, state=state)
    out = evaluate(value=out, state=state)
    out = evaluate(value=out, state=state)
    out = evaluate(value=out, state=state)
    return out