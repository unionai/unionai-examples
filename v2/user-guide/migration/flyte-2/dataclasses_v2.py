# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "learning_rate=0.1 n_estimators=100"
# ///

# {{docs-fragment all}}
from dataclasses import dataclass

import flyte

env = flyte.TaskEnvironment(name="dataclasses")


# Plain dataclasses work directly as task I/O -- no @dataclass_json mixin needed.
# Pydantic BaseModels work the same way.
@dataclass
class TrainingConfig:
    learning_rate: float
    n_estimators: int
    max_depth: int = 6


@env.task
def make_config(learning_rate: float, n_estimators: int) -> TrainingConfig:
    return TrainingConfig(learning_rate=learning_rate, n_estimators=n_estimators)


@env.task
def train(config: TrainingConfig) -> str:
    return (
        f"trained with lr={config.learning_rate}, "
        f"n_estimators={config.n_estimators}, max_depth={config.max_depth}"
    )


@env.task
def main(learning_rate: float, n_estimators: int) -> str:
    config = make_config(learning_rate, n_estimators)
    return train(config)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, learning_rate=0.1, n_estimators=100)
    print(r.name)
    print(r.url)
    r.wait()
