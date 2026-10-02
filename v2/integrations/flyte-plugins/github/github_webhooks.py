# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-github[app]>=2.10.6",
# ]
# ///
"""The GitHub webhook receiver: verify a delivery, launch a run, return.

This is an app, not a task. It does no work itself — it authenticates GitHub's
HMAC, normalizes the delivery into a `WebhookEvent`, and launches the tasks in
`github_tasks.py`. Keeping the two apart is what lets those tasks be run,
tested, and retried on their own.

Deploy it, then point GitHub at the payload URL the dashboard shows:

    python github_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.github import GitHubProvider, events

# One provider, one route at /webhook/github, one dashboard at /.
#
# GITHUB_WEBHOOK_SECRET is mounted for you from the provider's
# `default_secret_env`, so it does not need naming again in `secrets=`.
#
# `scopes` is an allowlist of repositories. A delivery from anywhere else is
# acknowledged — so GitHub stops retrying it — but never dispatched. So is a
# delivery carrying no repository at all: an allowlist cannot vouch for an
# event it cannot attribute.
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

    `run_once` is what makes this safe to call repeatedly. GitHub retries any
    non-2xx delivery and an operator may re-send one by hand; both arrive with
    the same `dedupe_key()`, and only the first launches a run. A *later* change
    to the same pull request gets its own key, because the key folds in the
    provider's own timestamp.

    Handlers must `await run_once.aio(...)` rather than call the blocking form:
    the blocking form stalls the app's event loop, and GitHub times a delivery
    out in ten seconds.
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
        # An earlier delivery of this same event already launched it.
        return {"skipped": result.run.name, "url": result.run.url}
    return {"run": result.run.name, "url": result.run.url}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    # The dashboard lists every provider's payload URL, whether its secret is
    # mounted, and how it is verified — paste the GitHub row into
    # Settings -> Webhooks -> Add webhook.
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
