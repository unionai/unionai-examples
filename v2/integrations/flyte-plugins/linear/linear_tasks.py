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
"""Tasks launched by the Linear webhook receiver.

`triage_issue` comments on an issue through Linear's GraphQL API, using `gql`.

`replay_sample_delivery` runs without a Linear workspace:

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
    """Comment on an issue."""
    import os

    from gql import Client, gql
    from gql.transport.httpx import HTTPXAsyncTransport

    transport = HTTPXAsyncTransport(
        url="https://api.linear.app/graphql",
        # Linear expects the API key with no "Bearer " prefix.
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
# A separate environment with no secrets, so the replay runs before any
# secret is created.
replay_env = flyte.TaskEnvironment(
    name="linear-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-linear"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the sample delivery bundled with the plugin."""
    import flyteplugins.linear as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    # Check the header name too. The sample's headers come from the plugin, so
    # the round trip above passes whatever the header is called. Linear sends
    # `Linear-Signature`, with no `X-` prefix.
    assert list(headers) == ["Linear-Signature"], f"unexpected signature header: {list(headers)}"
    assert not plugin.verify(body, {"X-Linear-Signature": headers["Linear-Signature"]}, secret), (
        "X-Linear-Signature must not verify"
    )

    event = plugin.parse(headers, body)
    return {
        # `Issue.create`: Linear sends the type and action separately.
        "qualified_type": event.qualified_type,
        "scope": event.scope or "",
        "title": event.title or "",
        "url": event.url or "",
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
