# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "step=1"
# ///

# {{docs-fragment all}}
from datetime import timedelta

import flyte

# Image, resources, and caching move to the TaskEnvironment, so they are declared
# once and shared by every task in the environment.
env = flyte.TaskEnvironment(
    name="training",
    image=flyte.Image.from_debian_base().with_pip_packages("scikit-learn", "pandas"),
    resources=flyte.Resources(cpu="2", memory="4Gi"),  # "memory", not "mem"
    cache="auto",
)


# retries and timeout stay on the task decorator.
@env.task(retries=3, timeout=timedelta(minutes=30))
def train_epoch(step: int) -> float:
    # A stand-in for a training step that returns the current loss.
    return 1.0 / (step + 1)


@env.task
def main(step: int) -> float:
    return train_epoch(step)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, step=1)
    print(r.name)
    print(r.url)
    r.wait()
