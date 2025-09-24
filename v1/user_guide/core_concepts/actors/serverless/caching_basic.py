from time import sleep

import union


actor = union.ActorEnvironment(
    name="my-actor",
    replica_count=1,
)


@union.actor_cache
def load_model(state: int) -> callable:
    sleep(4)  # simulate model loading
    return lambda value: state + value


@actor.task
def evaluate(value: int, state: int) -> int:
    model = load_model(state=state)
    return model(value)


@union.workflow
def wf(init_value: int = 1, state: int = 3) -> int:
    out = evaluate(value=init_value, state=state)
    out = evaluate(value=out, state=state)
    out = evaluate(value=out, state=state)
    out = evaluate(value=out, state=state)
    return out