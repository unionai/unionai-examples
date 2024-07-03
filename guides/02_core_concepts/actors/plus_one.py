from flytekit import workflow, Resources
from unionai.actor import ActorEnvironment

actor = ActorEnvironment(
    name="my_actor",
    replica_count=1,
    parallelism=1,
    backlog_length=50,
    ttl_seconds=300,
    requests=Resources(cpu="2", mem="500Mi"),
    container_image="ghcr.io/unionai/unionai-actors:0.1",
)


@actor
def plus_one(input: int) -> int:
    return input + 1


@workflow
def wf(input: int = 0) -> int:
    a = plus_one(input=input)
    b = plus_one(input=a)
    c = plus_one(input=b)
    return plus_one(input=c)
