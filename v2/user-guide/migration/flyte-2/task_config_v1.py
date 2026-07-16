from datetime import timedelta

import flytekit
from flytekit import Resources

image = flytekit.ImageSpec(
    name="training-image",
    packages=["scikit-learn", "pandas"],
)


@flytekit.task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
    retries=3,
    timeout=timedelta(minutes=30),
)
def train_epoch(step: int) -> float:
    # A stand-in for a training step that returns the current loss.
    return 1.0 / (step + 1)


@flytekit.workflow
def main(step: int) -> float:
    return train_epoch(step=step)
