from flytekit import task, workflow

@task
def foo_task2() -> str:
    return "Hello!"

@workflow
def foo_wf() -> str:
    res = foo_task2()
    return res