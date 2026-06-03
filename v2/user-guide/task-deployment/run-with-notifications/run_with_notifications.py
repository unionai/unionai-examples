# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "compute"
# params = ""
# ///

# {{docs-fragment run-with-notifications}}
import os
import flyte
from flyte import notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(name="notify_example")

SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
NOTIFICATION_EMAIL = os.environ["NOTIFICATION_EMAIL"]


@env.task
def compute(x: int, y: int) -> int:
    return x + y


if __name__ == "__main__":
    result = flyte.with_runcontext(
        notifications=(
            notify.Slack(
                on_phase=ActionPhase.SUCCEEDED,
                webhook_url=SLACK_WEBHOOK_URL,
                message="Run {{.Run.Name}} succeeded.",
            ),
            notify.Email(
                on_phase=ActionPhase.FAILED,
                recipients=[NOTIFICATION_EMAIL],
                subject="ALERT: Run {{.Run.Name}} failed",
                body="Run: {{.Run.Name}}\nError: {{.Error}}",
            ),
        ),
    ).run(compute, x=3, y=7)
    print(f"Result: {result}")
# {{/docs-fragment run-with-notifications}}
