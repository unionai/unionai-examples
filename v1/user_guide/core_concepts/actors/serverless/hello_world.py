import union


actor = union.ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=30,
    requests=union.Resources(
        cpu="2",
        mem="300Mi",
    ),
)


@actor.task
def say_hello() -> str:
    return "hello"


@union.workflow
def wf():
    say_hello()
