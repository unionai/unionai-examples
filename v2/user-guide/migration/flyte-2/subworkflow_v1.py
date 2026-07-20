from flytekit import task, workflow


@task
def impute(value: float) -> float:
    # Replace missing/negative sentinel values with 0.
    return value if value >= 0 else 0.0


@task
def scale(value: float) -> float:
    return value / 100.0


@workflow
def preprocess(value: float) -> float:
    imputed = impute(value=value)
    return scale(value=imputed)


@workflow
def main(raw_value: float) -> float:
    return preprocess(value=raw_value)
