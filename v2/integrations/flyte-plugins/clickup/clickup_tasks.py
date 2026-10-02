# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-clickup>=2.10.6",
#    "httpx>=0.27",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""The task a ClickUp webhook launches, over ClickUp's REST API.

ClickUp ships no official Python SDK, and its API is a handful of REST calls —
so `httpx` directly beats a thin third-party wrapper, and the plugin's job stops
at the webhook.

`replay_sample_delivery` is the entrypoint, and needs no ClickUp workspace:

    flyte run --local clickup_tasks.py replay_sample_delivery
"""

import flyte

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="clickup-ops",
    image=flyte.Image.from_debian_base().with_pip_packages("httpx"),
    secrets=[flyte.Secret(key="clickup-api-token", as_env_var="CLICKUP_API_TOKEN")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

CLICKUP_API = "https://api.clickup.com/api/v2"


@env.task
async def close_ticket(task_id: str) -> str:
    """Read a ticket, then close it only if it is not closed already.

    The pre-check is the point. ClickUp accepts a redundant status write, so
    without it a redelivered webhook would produce a second, misleading
    audit-log entry on the ticket — `run_once` keeps duplicate *runs* away, but
    an idempotent task is what keeps a re-run from lying.
    """
    import os

    import httpx

    headers = {"Authorization": os.environ["CLICKUP_API_TOKEN"]}
    async with httpx.AsyncClient(base_url=CLICKUP_API, headers=headers, timeout=15.0) as client:
        current = (await client.get(f"/task/{task_id}")).raise_for_status().json()
        if current["status"]["status"] == "complete":
            return "already complete"
        response = await client.put(f"/task/{task_id}", json={"status": "complete"})
        response.raise_for_status()
    return "closed"
# {{/docs-fragment task}}


# {{docs-fragment replay}}
# Its own environment, with no secrets: the replay needs none, and a task
# environment that declares a secret cannot start until that secret exists.
replay_env = flyte.TaskEnvironment(
    name="clickup-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-clickup"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the real delivery the plugin ships. No workspace needed."""
    import flyteplugins.clickup as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    event = plugin.parse(headers, body)
    return {
        # `taskCreated` — one string, because ClickUp sends no separate action.
        "qualified_type": event.qualified_type,
        "action_is_none": str(event.action is None),
        "scope": event.scope or "",
        "title": event.title or "",
        "dedupe_key": event.dedupe_key(),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
