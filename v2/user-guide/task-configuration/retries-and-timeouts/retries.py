# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment import-and-env}}
from datetime import timedelta

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="retries", resources=flyte.Resources(cpu=1, memory="250Mi"))
# {{/docs-fragment import-and-env}}


async def fetch_from_flaky_upstream() -> str:
    """Stand-in for a call to an unreliable external service."""
    return "ok"


# {{docs-fragment retry-count}}
@env.task(retries=3)
async def call_service() -> str:
    # retries=3 -> up to 4 attempts (1 original + 3 retries).
    # Each retry runs in a fresh pod, so nothing from the failed attempt carries over.
    return await fetch_from_flaky_upstream()
# {{/docs-fragment retry-count}}


# {{docs-fragment retry-backoff}}
@env.task(
    retries=flyte.RetryStrategy(
        count=5,
        backoff=flyte.Backoff(
            base=timedelta(seconds=10),  # first retry waits 10s
            factor=2.0,                  # then 20s, 40s, 80s, ...
            cap=timedelta(minutes=5),    # never wait longer than 5m between retries
        ),
    ),
)
async def call_flaky_api() -> str:
    # The delay before the n-th retry (0-indexed) is min(base * factor**n, cap).
    # Backoff gives a recovering downstream room to breathe instead of hammering it.
    return await fetch_from_flaky_upstream()
# {{/docs-fragment retry-backoff}}


# {{docs-fragment non-recoverable}}
@env.task(retries=3)
async def validate_and_process(x: int) -> str:
    if x < 0:
        # A negative input will never succeed, so don't waste the retry budget on it.
        # NonRecoverableError fails the action on attempt #1 — no retries are consumed.
        raise flyte.errors.NonRecoverableError(
            f"Input x={x} is negative — retrying will not help."
        )
    return f"processed({x})"
# {{/docs-fragment non-recoverable}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(validate_and_process, x=-5)
    print(run.name)
    print(run.url)
    run.wait()
# {{/docs-fragment run}}
