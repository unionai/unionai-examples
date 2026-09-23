# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte>=2.6.12",
# ]
# ///
"""The task that every trigger in this tutorial launches."""

from datetime import datetime, timezone

import flyte

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="event_driven",
    resources=flyte.Resources(cpu="1", memory="512Mi"),
)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@env.task
async def on_object(object_key: str = "", event_time: datetime = EPOCH) -> str:
    """Process one object. Stand in for your real pipeline."""
    return f"processed {object_key} on task version {flyte.ctx().version}"
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.deploy(env))
