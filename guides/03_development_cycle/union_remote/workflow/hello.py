from flytekit import task, workflow

@task
def hello_task() -> str:
    return "Hello!"

@workflow
def hello_wf() -> str:
    res = hello_task()
    return res