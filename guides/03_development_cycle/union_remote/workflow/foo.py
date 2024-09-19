from flytekit import task, workflow

@task
def foo_task() -> str:
    return "Hello!"

@workflow
def foo_wf() -> str:
    res = foo_task()
    return res