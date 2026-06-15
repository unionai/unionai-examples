# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
#     "httpx",
#     "fastapi",
#     "uvicorn",
# ]
# ///

"""Webhook-triggered agent — kick off the agent loop on external events.

A small FastAPI app (deployed via ``flyte.app``) exposes a ``POST /trigger``
endpoint. When an external service POSTs an event payload, the app launches a
fresh agent run as a durable Flyte task.

Deploy::

    flyte deploy webhook_agent.py webhook_env
"""

from __future__ import annotations

import os

import flyte
import flyte.app
from flyte.ai.agents import Agent

# Tools + agent live in a task environment.
tool_env = flyte.TaskEnvironment(
    name="webhook-agent-tools",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm", "httpx"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@tool_env.task
async def fetch_pr(repo: str, number: int) -> dict[str, str]:
    """Fetch review-relevant metadata for a specific GitHub pull request."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        resp = await client.get(f"https://api.github.com/repos/{repo}/pulls/{number}")
        resp.raise_for_status()
        pr = resp.json()
    return {
        "title": str(pr.get("title") or ""),
        "author": str((pr.get("user") or {}).get("login") or ""),
        "url": str(pr.get("html_url") or ""),
        "body": str(pr.get("body") or ""),
    }


@tool_env.task
async def post_comment(repo: str, number: int, comment: str) -> str:
    """Post a comment on a GitHub issue or PR (stub)."""
    flyte.logger.info("Would post on %s#%d: %s", repo, number, comment)
    return "ok"


agent = Agent(
    name="pr-reviewer",
    instructions=(
        "You are a code-review assistant. Given a webhook event for a pull "
        "request, fetch the PR metadata, summarize the change, and post a comment."
    ),
    model="claude-haiku-4-5",
    tools=[fetch_pr, post_comment],
    max_turns=10,
)


# {{docs-fragment webhook}}
@tool_env.task(report=True)
async def review_pr(repo: str, pr_number: int, event: str) -> str:
    """Durable task that runs the agent for a single webhook event."""
    message = f"GitHub webhook fired for {repo}#{pr_number} (event={event}). Review the PR."
    result = await agent.run.aio(message)
    return result.summary or result.error


def _build_app():
    from fastapi import FastAPI

    api = FastAPI(title="flyte-agent-webhook")

    @api.post("/trigger")
    async def trigger(payload: dict) -> dict[str, str]:
        repo = payload.get("repository")
        pr_number = int(payload.get("pull_request", {}).get("number", 0))
        event = payload.get("action")
        run = await flyte.run.aio(review_pr, repo=repo, pr_number=pr_number, event=event)
        return {"run_url": run.url, "name": run.name}

    return api


webhook_env = flyte.app.AppEnvironment(
    name="flyte-agent-webhook",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "litellm"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    depends_on=[tool_env],
)


@webhook_env.server
async def serve():
    import uvicorn

    config = uvicorn.Config(_build_app(), host="0.0.0.0", port=webhook_env.get_port().port)
    await uvicorn.Server(config).serve()
# {{/docs-fragment webhook}}


if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(webhook_env)
    print(f"Webhook URL: {deployments[0].envs['flyte-agent-webhook'].deployed_app.url}")
