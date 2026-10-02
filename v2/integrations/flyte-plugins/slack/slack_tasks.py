# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-slack>=2.10.6",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""Sending to Slack from tasks, and gating a run on a button click.

Receiving is the receiver's job; sending is a task's. This plugin is one of two
that carry more than a webhook provider, because two things here are not
reshaped JSON:

- `notify` replaces the fifty lines of `requests` and `ok`-checking every
  integration ends up hand-rolling.
- `approval` is the round trip — post buttons, park the run on a condition,
  resume when someone clicks. The condition is the part only Flyte can do.

`replay_sample_delivery` is the entrypoint, and needs no Slack workspace:

    flyte run --local slack_tasks.py replay_sample_delivery
"""

import flyte
from flyteplugins.slack import approval, notify

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="slack-bot",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    # The `xoxb-` credential from OAuth & Permissions. Posting needs the
    # `chat:write` scope, and the bot needs a `/invite` into the channel.
    secrets=[flyte.Secret(key="SLACK_BOT_TOKEN", as_env_var="SLACK_BOT_TOKEN")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def answer(channel: str, text: str, thread_ts: str) -> str:
    """Post a threaded reply, then edit it in place when the work finishes.

    `post` returns the message's `ts`, which is both the thread anchor and the
    address `update` edits — so a progress counter is two calls, not a second
    API surface to learn.
    """
    ts = await notify.post(channel, f"Working on: {text}", thread_ts=thread_ts)
    await notify.update(channel, ts, f"Done: {text}")
    return ts
# {{/docs-fragment task}}


# {{docs-fragment approval}}
@env.task
async def deploy_with_approval(release: str, channel: str = "C0DEPLOYS") -> str:
    """Ask Slack for a decision, and block until somebody clicks.

    `approval.request` posts Block Kit buttons and parks the run on a
    `flyte.new_condition`, then replaces the buttons with a "decided by" line so
    nobody clicks twice.

    The same condition is answerable from the Flyte UI, so an approval nobody
    clicks in Slack is not stuck: the run shows the same prompt, and either path
    resolves it.
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
# Its own environment, with no secrets: the replay needs none, and a task
# environment that declares a secret cannot start until that secret exists.
replay_env = flyte.TaskEnvironment(
    name="slack-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the real delivery the plugin ships. No workspace needed.

    Slack's sample signs at call time rather than carrying a fixed signature,
    because its verifier enforces a five-minute replay window — a delivery
    signed at a hard-coded timestamp would start failing the moment it aged out.
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
        # The replay window, in seconds. Slack is the one provider that has one.
        "max_request_age": str(plugin.MAX_REQUEST_AGE_SECONDS),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
