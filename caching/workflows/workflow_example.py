from flytekit import task, workflow, LaunchPlan

@task
def task_1(a: int, b: int, c: int) -> int:
    return a + b + c

@task
def task_2(m: int, n: int) -> int:
    return m * n

@task
def task_3(x: int, y: int) -> int:
    return x - y

@workflow
def my_workflow(a: int, b: int, c: int, m: int, n: int) -> int:
    x = task_1(a=a, b=b, c=c)
    y = task_2(m=m, n=n)
    return task_3(x=x, y=y)

LaunchPlan.get_or_create(
    workflow=my_workflow,
    name="my_workflow_custom_lp",
    fixed_inputs={"a": 6},
    default_inputs={"b": 4, "c": 5}
)
