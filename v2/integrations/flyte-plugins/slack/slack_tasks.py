# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-slack>=2.10.6",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""Tasks that post to Slack and wait for approval button clicks.

- `answer` posts a threaded reply with `notify.post`, then edits it with
  `notify.update`.
- `deploy_with_approval` uses `approval.request` to pause the run until
  someone clicks a button.

`replay_sample_delivery` runs without a Slack workspace:

    flyte run --local slack_tasks.py replay_sample_delivery
"""

import flyte
from flyteplugins.slack import approval, notify

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="slack-bot",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    # The bot token (xoxb-...) from OAuth & Permissions. Posting requires the
    # `chat:write` scope, and the bot must be invited to the channel.
    secrets=[flyte.Secret(key="SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def answer(channel: str, text: str, thread_ts: str) -> str:
    """Post a threaded reply, then edit it when the work finishes.

    `post` returns the message's `ts`, which `update` uses to edit it.
    """
    ts = await notify.post(channel, f"Working on: {text}", thread_ts=thread_ts)
    await notify.update(channel, ts, f"Done: {text}")
    return ts
# {{/docs-fragment task}}


# {{docs-fragment approval}}
@env.task
async def deploy_with_approval(release: str, channel: str = "C0DEPLOYS") -> str:
    """Post approval buttons and wait for a click.

    `approval.request` posts Block Kit buttons and waits on a
    `flyte.new_condition`. The handler added by `approval.register` resolves
    the condition when someone clicks. The condition can also be resolved
    from the Flyte UI.
    """
    decision = await approval.request.aio(
        channel,
        f"Deploy `{release}` to prod?",
        options=["approve", "reject"],
        timeout=3600,
    )
    if decision != "approve":
        return f"{release}: not deployed ({decision})"
    return f"{release}: deployed"
# {{/docs-fragment approval}}


# {{docs-fragment replay}}
# A separate environment with no secrets, so the replay runs before any
# secret is created.
replay_env = flyte.TaskEnvironment(
    name="slack-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the sample delivery bundled with the plugin.

    The provider rejects deliveries more than five minutes old, so
    `SAMPLE_DELIVERY` signs the payload with the current time when called.
    """
    import flyteplugins.slack as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    event = plugin.parse(headers, body)
    return {
        "qualified_type": event.qualified_type,
        "scope": event.scope or "",
        "title": event.title or "",
        "actor": event.actor or "",
        "dedupe_key": event.dedupe_key(),
        # Maximum delivery age, in seconds.
        "max_request_age": str(plugin.MAX_REQUEST_AGE_SECONDS),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
