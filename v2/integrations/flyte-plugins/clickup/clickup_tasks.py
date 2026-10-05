# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-clickup>=2.10.7",
#    "httpx>=0.27",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""Tasks launched by the ClickUp webhook receiver.

`close_ticket` sets a task's status through ClickUp's REST API, using `httpx`.

`replay_sample_delivery` runs without a ClickUp workspace:

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
    """Close a ticket unless it's already complete.

    `run_once` prevents duplicate runs, not duplicate writes within a run.
    ClickUp accepts a redundant status write and logs it, so the task checks
    the current status first.
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
# A separate environment with no secrets, so the replay runs before any
# secret is created.
replay_env = flyte.TaskEnvironment(
    name="clickup-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-clickup"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the sample delivery bundled with the plugin."""
    import flyteplugins.clickup as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    # Check the header name too. The sample's headers come from the plugin, so
    # the round trip above passes whatever the header is called. ClickUp sends
    # `X-Signature`, not `X-Clickup-Signature`.
    assert list(headers) == ["X-Signature"], f"unexpected signature header: {list(headers)}"
    assert not plugin.verify(body, {"X-Clickup-Signature": headers["X-Signature"]}, secret), (
        "X-Clickup-Signature must not verify"
    )

    event = plugin.parse(headers, body)
    return {
        # `taskCreated`: ClickUp sends a single event name with no action.
        "qualified_type": event.qualified_type,
        "action_is_none": str(event.action is None),
        "scope": event.scope or "",
        "title": event.title or "",
        "dedupe_key": event.dedupe_key(),
        # The header that carries the signature.
        "signature_header": next(iter(headers)),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
