from dataclasses import dataclass

from dataclasses_json import dataclass_json
from flytekit import task, workflow


@dataclass_json
@dataclass
class TrainingConfig:
    learning_rate: float
    n_estimators: int
    max_depth: int = 6


@task
def make_config(learning_rate: float, n_estimators: int) -> TrainingConfig:
    return TrainingConfig(learning_rate=learning_rate, n_estimators=n_estimators)


@task
def train(config: TrainingConfig) -> str:
    return (
        f"trained with lr={config.learning_rate}, "
        f"n_estimators={config.n_estimators}, max_depth={config.max_depth}"
    )


@workflow
def main(learning_rate: float, n_estimators: int) -> str:
    config = make_config(learning_rate=learning_rate, n_estimators=n_estimators)
    return train(config=config)
