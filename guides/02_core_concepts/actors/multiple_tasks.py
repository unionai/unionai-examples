from flytekit import current_context, workflow, LaunchPlan, Resources
from unionai.actor import ActorEnvironment

actor = ActorEnvironment(
    name="my_actor",
    replica_count=1,
    parallelism=1,
    backlog_length=50,
    ttl_seconds=30,
    requests=Resources(cpu="1", mem="450Mi")
)


@actor.task
def say_hello(name: str) -> str:
    return f"hello {name}"


@actor.task
def scream_hello(name: str) -> str:
    return f"HELLO {name}"


@workflow
def my_child_wf(name: str) -> str:
    return scream_hello(name=name)


my_child_wf_lp = LaunchPlan.get_default_launch_plan(current_context(),
                                                    my_child_wf)


@workflow
def my_parent_wf(name: str) -> str:
    a = say_hello(name=name)
    b = my_child_wf(name=a)
    return my_child_wf_lp(name=b)
