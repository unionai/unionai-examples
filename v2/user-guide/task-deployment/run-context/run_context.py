# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "root"
# params = ""
# ///

# {{docs-fragment basic}}
import flyte

env = flyte.TaskEnvironment("run-context-example")


@env.task
async def process(n: int) -> int:
    return n * 2


@env.task
async def root() -> int:
    return await process(21)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(
        name="my-run",
        project="my-project",
        domain="development",
    ).run(root)
# {{/docs-fragment basic}}


# {{docs-fragment read-ctx}}
@env.task
async def inspect_context() -> str:
    ctx = flyte.ctx()
    action = ctx.action
    return (
        f"run={action.run_name}, "
        f"action={action.name}, "
        f"mode={ctx.mode}, "
        f"in_cluster={ctx.is_in_cluster()}"
    )
# {{/docs-fragment read-ctx}}


# {{docs-fragment integration-naming}}
import wandb  # type: ignore[import]


@env.task
async def train_model(epochs: int) -> float:
    ctx = flyte.ctx()
    # Use run_name to tie the W&B run to this Flyte run
    run = wandb.init(
        project="my-project",
        name=ctx.action.run_name,
        config={"epochs": epochs},
    )
    # ... training logic ...
    loss = 0.42
    run.log({"loss": loss})
    run.finish()
    return loss
# {{/docs-fragment integration-naming}}


# {{docs-fragment raw-data-path}}
if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(
        # Store all task outputs in a dedicated S3 prefix for this run
        raw_data_path="s3://my-bucket/runs/experiment-42/",
    ).run(root)
# {{/docs-fragment raw-data-path}}

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
