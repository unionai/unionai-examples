# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-clickup[app]>=2.10.7",
# ]
# ///
"""ClickUp webhook receiver.

Deploy the app, then enter the payload URL from its dashboard in ClickUp under
Space Settings -> Integrations -> Webhooks:

    python clickup_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.clickup import ClickUpProvider, events

# CLICKUP_WEBHOOK_SECRET is mounted automatically.
#
# `scopes` lists ClickUp list IDs. The provider reads the list ID from both
# list events and task events.
app_env = WebhookAppEnvironment(
    name="clickup-webhooks",
    providers=[ClickUpProvider()],
    scopes=["9000"],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-clickup[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.Task.STATUS_UPDATED)
async def on_status_updated(event: WebhookEvent) -> dict:
    """Launch a run once per status change.

    `qualified_type` is `taskStatusUpdated`, and `action` is None.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="clickup-ops.close_ticket", auto_version="latest")
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        task_id=event.resource_id,
    )
    return {"run": result.run.name, "created": result.created}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
