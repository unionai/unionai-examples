# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-jira[app]>=2.10.6",
# ]
# ///
"""The Jira webhook receiver — the one provider that does not sign.

Jira Cloud sends no signature, so there is no HMAC to check. `JiraProvider`
authenticates with a shared token in an `X-Webhook-Token` header instead, and
reports `signed=False` so the dashboard says so plainly rather than implying a
guarantee that is absent.

Jira cannot send custom headers itself, so something in front of this app has to
inject that header — an API gateway, an ingress rule, or a Jira Automation rule
using *Send web request*, which can. Read the guide's authentication section
before exposing this route.

    python jira_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.jira import JiraProvider, events

# JIRA_WEBHOOK_TOKEN is mounted from the provider's `default_secret_env`. It is a
# shared token, not a signing secret: anything holding it can post a delivery,
# and the token travels on every request rather than signing one. So treat the
# proxy in front and the `scopes` allowlist below as part of the auth story, not
# as extras.
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

    `event.resource_id` is the issue key (`PROJ-1`). That is the stable handle
    the Jira API takes, and unlike the numeric id it is also what a human reads.
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
