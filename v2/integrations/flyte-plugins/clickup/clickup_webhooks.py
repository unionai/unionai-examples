# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-clickup[app]>=2.10.6",
# ]
# ///
"""The ClickUp webhook receiver.

Deploy it, then paste the payload URL the dashboard shows into
Space Settings -> Integrations -> Webhooks:

    python clickup_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.clickup import ClickUpProvider, events

# CLICKUP_WEBHOOK_SECRET is mounted from the provider's `default_secret_env`.
#
# `scopes` matches the ClickUp list id. The provider reads it from the top level
# on list-scoped events and from the nested task on task-scoped ones, so one
# allowlist attributes both.
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
    """Launch a run when a ticket changes status, once per change.

    ClickUp does not split type and action: the event name is one camelCase
    string, so `qualified_type` is `taskStatusUpdated` and `action` is None.
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
