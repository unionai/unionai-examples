# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-jira>=2.10.6",
#    "jira>=3.8",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""Tasks launched by the Jira webhook receiver.

`triage_issue` comments on an issue and transitions it, using the `jira` client.

`replay_sample_delivery` runs without a Jira site:

    flyte run --local jira_tasks.py replay_sample_delivery
"""

import flyte

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="jira-ops",
    image=flyte.Image.from_debian_base().with_pip_packages("jira"),
    secrets=[
        flyte.Secret(key="jira-url", as_env_var="JIRA_URL"),
        flyte.Secret(key="jira-email", as_env_var="JIRA_EMAIL"),
        flyte.Secret(key="jira-api-token", as_env_var="JIRA_API_TOKEN"),
    ],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def triage_issue(issue_key: str) -> str:
    """Comment on an issue and move it to In Progress.

    Transition IDs differ between workflows, so this looks up the transition
    by name instead of hard-coding an ID.
    """
    import asyncio
    import os

    from jira import JIRA

    def _work() -> str:
        client = JIRA(
            server=os.environ["JIRA_URL"],
            basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]),
        )
        issue = client.issue(issue_key)
        client.add_comment(issue, "Triaged by Flyte.")
        for transition in client.transitions(issue):
            if transition["name"].lower() == "in progress":
                client.transition_issue(issue, transition["id"])
                return transition["name"]
        return "no matching transition"

    # The `jira` client is synchronous; run it off the event loop.
    return await asyncio.to_thread(_work)
# {{/docs-fragment task}}


# {{docs-fragment replay}}
# A separate environment with no secrets, so the replay runs before any
# secret is created.
replay_env = flyte.TaskEnvironment(
    name="jira-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-jira"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the sample delivery bundled with the plugin.

    Jira doesn't sign deliveries, so `verify` compares a shared token in the
    `X-Webhook-Token` header. The sample's sign function only sets that header.
    """
    import flyteplugins.jira as plugin

    secret = "a-shared-webhook-token"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "the right token must verify"
    assert not plugin.verify(body, headers, "wrong-token"), "a wrong token must not"

    event = plugin.parse(headers, body)
    return {
        # For example, `jira:issue_created`. Some Jira event names have the
        # `jira:` prefix and some don't.
        "qualified_type": event.qualified_type,
        "scope": event.scope or "",
        "resource_id": event.resource_id or "",
        "title": event.title or "",
        "dedupe_key": event.dedupe_key(),
        # False for Jira. The setup dashboard shows this too.
        "provider_signs_deliveries": str(plugin.JiraProvider().signed),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
