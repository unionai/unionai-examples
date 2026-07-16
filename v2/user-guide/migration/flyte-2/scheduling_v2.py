# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
from datetime import datetime

import flyte

env = flyte.TaskEnvironment(name="scheduling")

# A Trigger replaces LaunchPlan + CronSchedule. It is attached directly to the
# task and deployed with it (flyte deploy). flyte.TriggerTime binds the
# scheduled fire time to a task input.
nightly_retrain = flyte.Trigger(
    name="nightly_retrain",
    automation=flyte.Cron("0 2 * * *"),  # 2 AM daily
    inputs={"trigger_time": flyte.TriggerTime},
    auto_activate=True,
)


@env.task(triggers=nightly_retrain)
def main(trigger_time: datetime = datetime(2024, 1, 1, 2, 0)) -> str:
    return f"retrained model at {trigger_time.isoformat()}"
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
