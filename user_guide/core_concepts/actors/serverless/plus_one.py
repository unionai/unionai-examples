from flytekit import workflow, Resources
from union.actor import ActorEnvironment

actor = ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=300,
    requests=Resources(cpu="2", mem="500Mi"),
)


@actor.task
def plus_one(input: int) -> int:
    return input + 1


@workflow
def wf(input: int = 0) -> int:
    a = plus_one(input=input)
    b = plus_one(input=a)
    c = plus_one(input=b)
    return plus_one(input=c)