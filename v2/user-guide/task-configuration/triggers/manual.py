# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "report_on_demand"
# params = ""
# ///

"""Triggers without automation: a named, pre-bound launch configuration.

Leave `automation` unset and the trigger becomes a saved way to run the task
(default inputs, queue, env vars, notifications) that nothing fires on its own.
It is fired on demand only, from the UI, the CLI, or `flyte.run` in Python.

Deploy it with:

    flyte deploy manual.py env
    flyte get trigger
"""

# {{docs-fragment manual-triggers}}
from datetime import datetime

import flyte
import flyte.notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(name="manual_trigger_example")

# No `automation=`: nothing schedules these. Each is a named set of inputs
# plus what to do when the run ends.
#
# Trigger inputs override the task's own defaults (`region="all"`, `days=7`
# below) for every run fired through the trigger. Inputs the trigger does not
# mention keep the task default, so `quick-report` still gets `as_of=None`.
quick_report = flyte.Trigger(
    name="quick-report",
    inputs={"region": "us-east", "days": 1},
    description="Yesterday only, for a fast sanity check",
    notifications=flyte.notify.Slack(
        on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        message=":x: quick-report {{.Run.Name}} ended in {{.Phase}}: {{.Error}}",
    ),
)

full_report = flyte.Trigger(
    name="full-report",
    inputs={"region": "all", "days": 30},
    description="The full monthly report",
    env_vars={"REPORT_VERBOSE": "1"},
    notifications=(
        flyte.notify.Email(
            on_phase=ActionPhase.SUCCEEDED,
            recipients=["reports@example.com"],
            subject="Monthly report {{.Run.Name}} is ready",
            body="Run: {{.Run.Name}}\nProject/Domain: {{.Run.Project}}/{{.Run.Domain}}",
        ),
        flyte.notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            message=":rotating_light: full-report {{.Run.Name}} failed: {{.Error}}",
        ),
    ),
)
# {{/docs-fragment manual-triggers}}

# {{docs-fragment manual-alongside-scheduled}}
# A scheduled trigger can sit next to the manual ones on the same task. Only a
# schedule can bind `flyte.TriggerTime`, since a manual trigger has no fire time.
nightly = flyte.Trigger(
    name="nightly",
    automation=flyte.Cron("0 2 * * *"),
    inputs={"as_of": flyte.TriggerTime, "region": "all", "days": 1},
)


@env.task(triggers=(quick_report, full_report, nightly))
async def report_on_demand(region: str = "all", days: int = 7, as_of: datetime | None = None) -> str:
    as_of = as_of or datetime.now()
    msg = f"report for region={region!r} over the last {days} day(s), as of {as_of.isoformat()}"
    print(msg)
    return msg
# {{/docs-fragment manual-alongside-scheduled}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
# {{/docs-fragment deploy}}
