from flytekit import task, workflow, LaunchPlan


@task
def t1(a: int, b: int, c: int) -> int:
    return a + b + c


@workflow
def sub_wf(a: int, b: int, c: int) -> int:
    return t1(a=a, b=b, c=c)


sub_wf_lp = LaunchPlan.get_or_create(sub_wf)


@workflow
def wf_cached(i: int = 0):
    sub_wf_lp(a=i, b=1, c=2).with_overrides(cache=True, cache_version="1.0")
    sub_wf(a=i, b=3, c=4).with_overrides(cache=True, cache_version="1.0")


@workflow
def wf_uncached(i: int = 0):
    sub_wf_lp(a=i, b=1, c=2)
    sub_wf(a=i, b=3, c=4)
