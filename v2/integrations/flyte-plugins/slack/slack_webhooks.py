# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-slack[app]>=2.10.6",
# ]
# ///
"""Slack webhook receiver for Events API callbacks, interactivity, and slash commands.

`SlackProvider` verifies and parses all three delivery types on one route;
`on_event` selects between them.

Deploy the app, then enter the payload URL from its dashboard at
api.slack.com/apps under Event Subscriptions, Interactivity & Shortcuts, and
each slash command:

    python slack_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.slack import SlackProvider, approval, events, notify

# SLACK_SIGNING_SECRET is mounted automatically. It's the signing secret from
# Basic Information, not the bot token, which goes on the task environment.
#
# `scopes` lists the channel IDs to act on: where the bot is mentioned, where
# /deploy is used, and where approvals are posted. Events from other channels
# are acknowledged but not dispatched.
app_env = WebhookAppEnvironment(
    name="slack-webhooks",
    providers=[SlackProvider()],
    scopes=["C0DEPLOYS"],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

# Adds a handler that resolves the condition behind each button posted by
# `approval.request`. Each button's value carries the run, action, and
# condition names, so no other configuration is needed.
approval.register(app_env)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.AppMention.ANY)
async def on_mention(event: WebhookEvent) -> dict:
    """Launch a run for each @-mention.

    The dedupe key identifies one message. For one run per thread, build a
    key from `thread_ts` and pass it to `run_once` instead.
    """
    import flyte.remote as remote

    task = remote.Task.get(name="slack-bot.answer", auto_version="latest")
    slack_event = event.payload["event"]
    result = await run_once.aio(
        task,
        key=event.dedupe_key(),
        channel=event.scope,
        text=event.title or "",
        thread_ts=slack_event.get("thread_ts") or slack_event["ts"],
    )
    return {"run": result.run.name, "created": result.created}


@app_env.on_event(events.Command, action="/deploy")
async def on_deploy_command(event: WebhookEvent) -> dict:
    """Acknowledge the /deploy slash command.

    `respond` posts to the command's `response_url` and needs no bot token.
    Slack expects a reply within three seconds, so acknowledge here and do
    longer work in a launched run.
    """
    await notify.respond(event.payload["response_url"], "Deploy queued.")
    return {"ok": True}
# {{/docs-fragment handler}}


# {{docs-fragment serve}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Setup dashboard: {deployment.url}")
# {{/docs-fragment serve}}
