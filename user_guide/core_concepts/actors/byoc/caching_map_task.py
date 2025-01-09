from functools import partial
from pathlib import Path
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
    replica_count=2,
)


class MyModel:
    """Simple model that multiples value with model_state."""

    def __init__(self, model_state: int):
        self.model_state = model_state

    def __call__(self, value: int):
        return self.model_state * value


@union.task(container_image=image, cache=True, cache_version="v1")
def create_model_state() -> union.FlyteFile:
    working_dir = Path(union.current_context().working_directory)
    model_state_path = working_dir / "model_state.txt"
    model_state_path.write_text("4")
    return model_state_path


@union.actor_cache
def load_model(model_state_path: union.FlyteFile) -> MyModel:
    # Simulate model loading time. This can take a long time
    # because the FlyteFile download is large, or when the  
    # model is loaded onto the GPU.
    sleep(10)
    with model_state_path.open("r") as f:
        model_state = int(f.read())

    return MyModel(model_state=model_state)


@actor.task
def inference(value: int, model_state_path: union.FlyteFile) -> int:
    model = load_model(model_state_path)
    return model(value)


@union.workflow
def run_inference(values: list[int] = list(range(20))) -> list[int]:
    model_state = create_model_state()
    inference_ = partial(inference, model_state_path=model_state)
    return union.map_task(inference_)(value=values)