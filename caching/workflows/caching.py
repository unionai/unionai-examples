from flytekit import task, workflow, LaunchPlan


@task
def t1(a: int, b: int, c: int) -> int:
    return a + b + c


@task
def t2(m: int, n: int) -> int:
    return m * n


@task
def t3(x: int, y: int) -> int:
    return x - y


@workflow
def wf(a: int, b: int, c: int, m: int, n: int) -> int:
    x = t1(a=a, b=b, c=c)
    y = t2(m=m, n=n)
    return t3(x=x, y=y)


LaunchPlan.get_or_create(
    workflow=wf,
    name="wf_custom_lp",
    fixed_inputs={"a": 6},
    default_inputs={"b": 4, "c": 5}
)
