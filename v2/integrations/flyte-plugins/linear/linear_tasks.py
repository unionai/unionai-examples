# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-linear>=2.10.7",
#    "gql[httpx]>=3.5",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""The task a Linear webhook launches, over Linear's GraphQL API.

Linear ships no official Python SDK and does not need one: its API is a single
GraphQL endpoint, so `gql` is the maintained client and the task below calls it
directly. The plugin's job stops at the webhook.

`replay_sample_delivery` is the entrypoint, and needs no Linear workspace:

    flyte run --local linear_tasks.py replay_sample_delivery
"""

import flyte

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="linear-triage",
    image=flyte.Image.from_debian_base().with_pip_packages("gql[httpx]"),
    secrets=[flyte.Secret(key="linear-api-key", as_env_var="LINEAR_API_KEY")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def triage_issue(issue_id: str, title: str) -> str:
    """Comment on an issue. Plain `gql` against Linear's one GraphQL endpoint."""
    import os

    from gql import Client, gql
    from gql.transport.httpx import HTTPXAsyncTransport

    transport = HTTPXAsyncTransport(
        url="https://api.linear.app/graphql",
        # Linear takes the API key raw, with no "Bearer " prefix.
        headers={"Authorization": os.environ["LINEAR_API_KEY"]},
    )
    mutation = gql(
        """
        mutation Comment($issueId: String!, $body: String!) {
          commentCreate(input: {issueId: $issueId, body: $body}) { success }
        }
        """
    )
    async with Client(transport=transport) as session:
        result = await session.execute(
            mutation,
            variable_values={"issueId": issue_id, "body": f"Triaged by Flyte: {title}"},
        )
    return str(result["commentCreate"]["success"])
# {{/docs-fragment task}}


# {{docs-fragment replay}}
# Its own environment, with no secrets: the replay needs none, and a task
# environment that declares a secret cannot start until that secret exists.
replay_env = flyte.TaskEnvironment(
    name="linear-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-linear"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the real delivery the plugin ships. No workspace needed."""
    import flyteplugins.linear as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    # The wire contract, which the round trip above cannot check: `verify` and
    # `SAMPLE_DELIVERY` agree with each other whatever the header is called, so
    # a wrong name passes conformance and then rejects every real delivery.
    # Linear signs with `Linear-Signature` -- note the missing `X-` prefix,
    # which looks like a typo and is not one.
    assert list(headers) == ["Linear-Signature"], f"unexpected signature header: {list(headers)}"
    assert not plugin.verify(body, {"X-Linear-Signature": headers["Linear-Signature"]}, secret), (
        "the old, wrong header name must not verify"
    )

    event = plugin.parse(headers, body)
    return {
        # `Issue.create` — Linear is one of the providers that splits the two.
        "qualified_type": event.qualified_type,
        "scope": event.scope or "",
        "title": event.title or "",
        "url": event.url or "",
        "dedupe_key": event.dedupe_key(),
        # The header a real delivery carries the signature in.
        "signature_header": next(iter(headers)),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
