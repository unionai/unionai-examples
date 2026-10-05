# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-jira[app]>=2.10.6",
# ]
# ///
"""Jira webhook receiver.

Jira Cloud doesn't sign deliveries. `JiraProvider` checks a shared token in the
`X-Webhook-Token` header instead, and reports `signed=False` on the dashboard.

Jira webhooks can't set custom headers, so an API gateway, an ingress rule, or
a Jira Automation rule using Send web request must add the header. See the
Authentication section of the Jira integration guide before exposing this route.

    python jira_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.jira import JiraProvider, events

# JIRA_WEBHOOK_TOKEN is mounted automatically. It's a shared token, sent with
# every request: anyone who has it can post a delivery. Restrict `scopes` to
# the projects the app should act on.
app_env = WebhookAppEnvironment(
    name="jira-webhooks",
    providers=[JiraProvider()],
    # Jira project keys.
    scopes=["PROJ"],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-jira[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.Issue.CREATED)
async def on_issue_created(event: WebhookEvent) -> dict:
    """Launch triage once per new issue.

    `event.resource_id` is the issue key, such as `PROJ-1`.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="jira-ops.triage_issue", auto_version="latest")
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        issue_key=event.resource_id,
    )
    return {"run": result.run.name, "created": result.created}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
