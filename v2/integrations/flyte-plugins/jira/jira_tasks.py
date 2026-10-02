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
"""The task a Jira webhook launches, over the `jira` client.

`replay_sample_delivery` is the entrypoint, and needs no Jira site:

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
    """Comment on an issue and move it to In Progress. Plain `jira` client.

    Transitions are named per workflow, not globally, so this resolves the name
    to an id rather than hard-coding one — a hard-coded id breaks the first time
    somebody edits the project's workflow.
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

    # The `jira` client is synchronous; keep it off the event loop.
    return await asyncio.to_thread(_work)
# {{/docs-fragment task}}


# {{docs-fragment replay}}
# Its own environment, with no secrets: the replay needs none, and a task
# environment that declares a secret cannot start until that secret exists.
replay_env = flyte.TaskEnvironment(
    name="jira-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-jira"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the real delivery the plugin ships. No Jira site needed.

    Note what `verify` does here: it compares a shared token in constant time,
    rather than checking a signature. The sample's "sign" function only sets the
    header, because there is nothing to sign.
    """
    import flyteplugins.jira as plugin

    secret = "a-shared-webhook-token"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "the right token must verify"
    assert not plugin.verify(body, headers, "wrong-token"), "a wrong token must not"

    event = plugin.parse(headers, body)
    return {
        # `jira:issue_created` — Jira namespaces some, but not all, event names.
        "qualified_type": event.qualified_type,
        "scope": event.scope or "",
        "resource_id": event.resource_id or "",
        "title": event.title or "",
        "dedupe_key": event.dedupe_key(),
        # False — and the setup dashboard says so.
        "provider_signs_deliveries": str(plugin.JiraProvider().signed),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
