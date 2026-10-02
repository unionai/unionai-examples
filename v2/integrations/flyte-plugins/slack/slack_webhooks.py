# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-slack[app]>=2.10.6",
# ]
# ///
"""The Slack webhook receiver: Events API, interactivity, and slash commands.

Slack is the broadest provider in this family because it delivers three
different shapes to the same route — event callbacks as JSON, interactivity
payloads and slash commands as form bodies. One `SlackProvider()` verifies and
normalizes all three; `on_event` is what tells them apart.

Deploy it, then paste the payload URL the dashboard shows into all three fields
at api.slack.com/apps (Event Subscriptions, Interactivity, and each slash
command):

    python slack_webhooks.py
"""

import pathlib

# {{docs-fragment app}}
import flyte
from flyte.extras.webhooks import WebhookAppEnvironment, WebhookEvent, run_once
from flyteplugins.slack import SlackProvider, approval, events, notify

# SLACK_SIGNING_SECRET is mounted from the provider's `default_secret_env`. That
# is the *signing secret* under Basic Information — not the `xoxb-` bot token
# that `notify` sends with, which belongs on a task environment instead.
app_env = WebhookAppEnvironment(
    name="slack-webhooks",
    providers=[SlackProvider()],
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-slack[app]"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

# One line, and every approval button posted by `approval.request` is answered
# from here on: the handler reads the run, action, and condition names off the
# button's own `value`, looks the condition up, and signals it. No configuration,
# because the button carries everything needed to answer it.
approval.register(app_env)
# {{/docs-fragment app}}


# {{docs-fragment handler}}
@app_env.on_event(events.AppMention.ANY)
async def on_mention(event: WebhookEvent) -> dict:
    """Answer an @-mention by launching a run, once per message.

    Slack's dedupe key is per message. To collapse a whole thread onto one run,
    build your own key from `thread_ts` and pass that to `run_once` instead.
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
    """A slash command. `respond` needs no token at all.

    It posts to the `response_url` every interaction and slash command carries,
    which makes it the zero-setup way to answer the click that launched you.
    Slack only shows a synchronous reply if it arrives within three seconds, so
    acknowledge here and let the launched run post the real answer.
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
