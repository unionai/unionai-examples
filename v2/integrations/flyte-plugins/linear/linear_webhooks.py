# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-linear[app]>=2.10.7",
# ]
# ///
"""Linear webhook receiver.

Deploy the app, then enter the payload URL from its dashboard in Linear under
Settings -> API -> Webhooks:

    python linear_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.linear import LinearProvider, events

# LINEAR_WEBHOOK_SECRET is mounted automatically.
#
# `scopes` lists Linear team IDs. For Comment and Reaction events, the provider
# reads the team ID from the related issue.
app_env = WebhookAppEnvironment(
    name="linear-webhooks",
    providers=[LinearProvider()],
    scopes=["team-000"],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-linear[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.Issue.CREATE)
async def on_issue_created(event: WebhookEvent) -> dict:
    """Launch triage once per new issue.

    The constant `Issue.CREATE` matches the `qualified_type` `Issue.create`.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="linear-triage.triage_issue", auto_version="latest")
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        issue_id=event.resource_id,
        title=event.title or "",
    )
    return {"run": result.run.name, "created": result.created}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
