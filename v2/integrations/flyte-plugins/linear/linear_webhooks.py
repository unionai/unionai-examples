# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-linear[app]>=2.10.7",
# ]
# ///
"""The Linear webhook receiver.

Deploy it, then paste the payload URL the dashboard shows into
Linear Settings -> API -> Webhooks:

    python linear_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.linear import LinearProvider, events

# LINEAR_WEBHOOK_SECRET is mounted from the provider's `default_secret_env`.
#
# `scopes` matches Linear's team id, which is what the provider puts in
# `WebhookEvent.scope` — including on Comment and Reaction payloads, where the
# team id is nested on the issue rather than sent at the top level. Without that
# fallback an allowlist would drop every non-Issue event as unattributable.
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

    Linear splits type and action, so the constant is `Issue.CREATE` and the
    normalized `qualified_type` reads `Issue.create`.
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
