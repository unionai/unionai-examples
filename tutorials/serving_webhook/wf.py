from flytekit import task, workflow


@task
def add_one(x: int) -> int:
    return x + 1


@workflow
def add_one_wf(x: int) -> int:
    return add_one(x=x)
