# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-github[app]>=2.10.6",
# ]
# ///
"""GitHub webhook receiver.

An app that verifies GitHub deliveries, parses them into `WebhookEvent`s, and
launches the tasks in `github_tasks.py`. Deploy those tasks first.

Deploy the app, then set the payload URL from its dashboard in GitHub:

    python github_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.github import GitHubProvider, events

# Serves the receiver at /webhook/github and a setup dashboard at /.
# GITHUB_WEBHOOK_SECRET is mounted automatically.
#
# `scopes` lists the repositories to act on. Deliveries from other
# repositories, or with no repository, are acknowledged but not dispatched.
app_env = WebhookAppEnvironment(
    name="github-webhooks",
    providers=[GitHubProvider()],
    scopes=["octo/repo"],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.PullRequest.OPENED)
async def on_pull_request_opened(event: WebhookEvent) -> dict:
    """Launch triage once per pull request.

    Retried and resent deliveries have the same `dedupe_key()`, so `run_once`
    launches only one run for them. Use `await run_once.aio(...)`: the blocking
    form stalls the app's event loop, and GitHub times out a delivery after
    ten seconds.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="github-triage.triage_pr", auto_version="latest")
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        repo=event.scope,
        number=event.payload["pull_request"]["number"],
    )
    if not result.created:
        # An earlier delivery of this event already launched a run.
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name, "url": result.run.url}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    # The dashboard shows the payload URL to enter in GitHub under
    # Settings -> Webhooks -> Add webhook.
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
