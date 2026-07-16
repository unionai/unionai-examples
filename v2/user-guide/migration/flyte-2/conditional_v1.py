from flytekit import task, workflow, conditional


@task
def train_gradient_boosting(n_rows: int) -> str:
    return f"trained gradient boosting on {n_rows} rows"


@task
def train_logistic_regression(n_rows: int) -> str:
    return f"trained logistic regression on {n_rows} rows"


@workflow
def main(n_rows: int) -> str:
    # Pick the model based on dataset size.
    return (
        conditional("model_choice")
        .if_(n_rows > 10_000)
        .then(train_gradient_boosting(n_rows=n_rows))
        .else_()
        .then(train_logistic_regression(n_rows=n_rows))
    )
