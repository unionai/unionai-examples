# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
#     "httpx",
# ]
# ///

"""Scheduled triage agent — wakes up daily and runs durable Flyte tasks as tools.

The "wakeup" is a regular Flyte task — the agent loop runs inside it, so every
tool call is durable, observable, and retryable on its own.

Deploy::

    flyte deploy scheduled_triage_agent.py env
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import flyte
from flyte.ai.agents import Agent

env = flyte.TaskEnvironment(
    name="triage-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("httpx", "litellm"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def list_open_issues(repo: str, max_count: int = 25) -> list[dict[str, str]]:
    """Fetch open issues for a GitHub repository."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        resp = await client.get(
            "https://api.github.com/search/issues",
            params={"q": f"repo:{repo} is:issue is:open", "per_page": min(max_count, 100)},
        )
        resp.raise_for_status()
        issues = resp.json().get("items", [])
    return [{"number": str(i["number"]), "title": i["title"], "url": i["html_url"]} for i in issues]


@env.task
async def classify_issue(title: str, body: str = "") -> Literal["urgent", "important", "normal", "noise"]:
    """Score one issue by severity using simple heuristics."""
    text = f"{title} {body}".lower()
    if any(kw in text for kw in ("critical", "outage", "data loss", "security", "p0")):
        return "urgent"
    if any(kw in text for kw in ("bug", "regression", "broken", "p1", "p2")):
        return "important"
    if any(kw in text for kw in ("typo", "nit", "tracking")):
        return "noise"
    return "normal"


@env.task
async def post_digest(channel: str, summary: str) -> dict[str, str]:
    """Post a Markdown digest to a chat channel (stub — replace with Slack/Linear/etc.)."""
    flyte.logger.info("Posting digest to %s:\n%s", channel, summary)
    return {"channel": channel, "delivered_at": datetime.utcnow().isoformat() + "Z"}


# {{docs-fragment scheduled}}
agent = Agent(
    name="github-triage",
    instructions=(
        "You are a GitHub issue triager. For each wakeup: list open issues for "
        "the configured repo, classify each one, group them by severity, and "
        "post a concise digest to the team channel. Always end by calling post_digest."
    ),
    model="claude-haiku-4-5",
    tools=[list_open_issues, classify_issue, post_digest],
    max_turns=20,
)


@env.task(
    triggers=flyte.Trigger(
        "daily-triage",
        flyte.Cron("0 9 * * *"),  # every day at 09:00
        inputs={"trigger_time": flyte.TriggerTime, "repo": "flyteorg/flyte", "channel": "#flyte-triage"},
    ),
    report=True,
)
async def triage_repo(trigger_time: datetime, repo: str, channel: str) -> str:
    """Scheduled wakeup that runs the triage agent end-to-end."""
    message = f"It is {trigger_time.isoformat()}. Triage the open issues in {repo} and post a digest to {channel}."
    with flyte.group("triage-loop"):
        result = await agent.run.aio(message)
    return result.summary or f"[triage failed] {result.error}"
# {{/docs-fragment scheduled}}


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
