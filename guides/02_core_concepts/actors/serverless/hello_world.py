from flytekit import workflow, Resources
from union.actor import ActorEnvironment

actor = ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=30,
    requests=Resources(
        cpu="2",
        mem="300Mi",
    ),
)


@actor.task
def say_hello() -> str:
    return "hello"


@workflow
def wf():
    say_hello()
