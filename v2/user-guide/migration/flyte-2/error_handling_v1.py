from flytekit import task, workflow


@task
def train_fold(max_depth: int) -> float:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    # Return validation accuracy for this hyperparameter.
    return 0.90 + 0.001 * max_depth


@task
def notify_failure() -> None:
    print("training run failed -- sending alert")


# The on_failure handler runs if any node in the workflow fails. There is no
# try/except inside a Flyte 1 workflow.
@workflow(on_failure=notify_failure)
def main(max_depth: int) -> float:
    return train_fold(max_depth=max_depth)
