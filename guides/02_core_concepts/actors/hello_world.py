from flytekit import workflow, Resources
from unionai.actor import ActorEnvironment

actor = ActorEnvironment(
    name="my_actor",
    replica_count=1,
    parallelism=1,
    backlog_length=10,
    ttl_seconds=30,
    requests=Resources(
        cpu="2",
        mem="300Mi",
    ),
)


@actor
def say_hello() -> str:
    return "hello"


@workflow
def wf():
    say_hello()
