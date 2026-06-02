# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment import-and-env}}
import asyncio
from datetime import timedelta

import flyte
from flyte import Timeout

env = flyte.TaskEnvironment(name="timeouts", resources=flyte.Resources(cpu=1, memory="250Mi"))
# {{/docs-fragment import-and-env}}


# {{docs-fragment max-runtime}}
@env.task(timeout=Timeout(max_runtime=timedelta(minutes=30)))
async def train_model() -> str:
    # max_runtime bounds the RUNNING phase of a single attempt. If the task is
    # still running after 30 minutes, this attempt is reaped as TIMED_OUT.
    # The budget is per-attempt: it resets fresh on every retry.
    ...
    return "model trained"
# {{/docs-fragment max-runtime}}


# {{docs-fragment max-queued-time}}
@env.task(timeout=Timeout(max_queued_time=timedelta(minutes=15)))
async def needs_scarce_gpu() -> str:
    # max_queued_time bounds the time spent waiting to run (QUEUED +
    # WAITING_FOR_RESOURCES). If the cluster can't find capacity within 15
    # minutes, fail fast instead of stalling indefinitely. Per-attempt.
    ...
    return "done"
# {{/docs-fragment max-queued-time}}


# {{docs-fragment deadline}}
@env.task(timeout=Timeout(deadline=timedelta(hours=2)))
async def must_finish_by() -> str:
    # deadline is an absolute wall-clock budget across ALL attempts, measured
    # from the first time the action was enqueued. Once 2 hours elapse, the
    # action is reaped no matter which phase it is in or how many retries remain.
    ...
    return "done"
# {{/docs-fragment deadline}}


# {{docs-fragment all-bounds}}
@env.task(
    timeout=Timeout(
        max_runtime=timedelta(minutes=30),      # per attempt, RUNNING only
        max_queued_time=timedelta(minutes=15),  # per attempt, waiting to run
        deadline=timedelta(hours=2),            # absolute, across all attempts
    ),
)
async def fully_bounded() -> str:
    ...
    return "done"
# {{/docs-fragment all-bounds}}


# {{docs-fragment timeout-shorthand}}
# A bare int (seconds) or timedelta is shorthand for Timeout(max_runtime=...).
@env.task(timeout=timedelta(minutes=30))
async def runtime_only() -> str:
    ...
    return "done"
# {{/docs-fragment timeout-shorthand}}


# {{docs-fragment retries-and-deadline}}
@env.task(
    retries=flyte.RetryStrategy(
        count=5,
        backoff=flyte.Backoff(base=timedelta(seconds=30), factor=2.0, cap=timedelta(minutes=5)),
    ),
    timeout=Timeout(
        max_runtime=timedelta(minutes=10),  # cap any single attempt
        deadline=timedelta(hours=1),        # but never spend more than 1h total
    ),
)
async def resilient_work() -> str:
    # Retries continue until either the retry budget is exhausted OR the 1h
    # deadline fires — whichever comes first. The deadline wins ties.
    ...
    return "done"
# {{/docs-fragment retries-and-deadline}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(fully_bounded)
    print(run.name)
    print(run.url)
    run.wait()
# {{/docs-fragment run}}
