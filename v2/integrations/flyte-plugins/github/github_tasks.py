# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.10.6",
#    "flyteplugins-github[review,auth]>=2.10.6",
#    "PyGithub>=2",
# ]
# main = "replay_sample_delivery"
# params = ""
# ///
"""The tasks a GitHub webhook launches — plus the one Flyte beats PyGithub at.

Three things live here:

- `triage_pr` calls PyGithub directly. There is deliberately no Flyte wrapper
  around the GitHub API: PyGithub is maintained by people who get deprecation
  notices first, and a task is just a function, so a wrapper would only add a
  surface to keep in sync with someone else's release calendar.
- `gated_merge` uses `review_pr`, which *is* the plugin's job: it parks the run
  on a `flyte.new_condition` until a human answers. The condition is the part
  only Flyte can do.
- `clone_at_head` mints a short-lived GitHub App token instead of holding a
  personal access token.

`replay_sample_delivery` is the entrypoint, and runs with no GitHub account, no
webhook, and no credentials:

    flyte run --local github_tasks.py replay_sample_delivery
"""

import flyte

# {{docs-fragment task}}
env = flyte.TaskEnvironment(
    name="github-triage",
    image=flyte.Image.from_debian_base().with_pip_packages("PyGithub"),
    # The PR-reading token. Separate from the webhook signing secret, which
    # belongs to the receiver app and never reaches a task.
    secrets=[flyte.Secret(key="github-token", as_env_var="GITHUB_TOKEN")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@env.task
async def triage_pr(repo: str, number: int) -> str:
    """Label a pull request by size. Plain PyGithub, called from a task."""
    import os

    from github import Auth, Github

    client = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
    pull = client.get_repo(repo).get_pull(number)
    changed = pull.additions + pull.deletions
    label = "size/s" if changed < 50 else "size/m" if changed < 500 else "size/l"
    pull.add_to_labels(label)
    return label
# {{/docs-fragment task}}


# {{docs-fragment review-gate}}
review_env = flyte.TaskEnvironment(
    name="github-review",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github[review]"),
    secrets=[flyte.Secret(key="github-token", as_env_var="GITHUB_TOKEN")],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@review_env.task
async def gated_merge(repo: str, number: int) -> str:
    """Park the run until a human answers, then branch on a typed decision.

    `review_pr` collects the pull request's metadata, raises a condition
    carrying it as JSON, and waits. The reviewer answers in the Flyte UI. The run
    survives restarts while it waits, because the condition is durable state on
    the backend rather than a held-open process.
    """
    from flyteplugins.github import review_pr

    decision = await review_pr(repo, number, instructions="Block on missing tests.")
    if decision.is_approved:
        return f"approved by {decision.reviewer}: {decision.summary}"
    blocking = ", ".join(c.path for c in decision.blocking_comments)
    return f"{decision.verdict}: {decision.summary} ({blocking})"
# {{/docs-fragment review-gate}}


# {{docs-fragment app-token}}
agent_env = flyte.TaskEnvironment(
    name="github-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github[auth]"),
    # A kebab-case secret key upper-cases into the environment variable
    # `mint_installation_token` reads, so no `as_env_var=` is needed.
    secrets=[
        flyte.Secret(key="github-app-id"),
        flyte.Secret(key="github-app-installation-id"),
        flyte.Secret(key="github-app-private-key"),
    ],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@agent_env.task
async def clone_at_head(repo: str) -> str:
    """Mint a one-hour App token per operation instead of storing a PAT.

    A token this short-lived is plenty for a clone or a `gh pr create`, and
    useless to anyone who later finds it in a log.
    """
    import asyncio

    from flyteplugins.github import clone_url, mint_installation_token

    # Synchronous, one HTTPS round trip — keep it off the event loop.
    token = await asyncio.to_thread(mint_installation_token)
    url = clone_url(repo, token)
    # Never return or log the token itself.
    return url.replace(token, "***") if token else url
# {{/docs-fragment app-token}}


# {{docs-fragment replay}}
# Its own environment, with no secrets: the replay needs none, and a task
# environment that declares a secret cannot start until that secret exists.
replay_env = flyte.TaskEnvironment(
    name="github-replay",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-github"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@replay_env.task
async def replay_sample_delivery() -> dict[str, str]:
    """Verify and parse the real delivery the plugin ships. No account needed.

    Every provider plugin exports a `SAMPLE_DELIVERY`: a trimmed but real
    payload, plus a function that signs it. It is what the plugin's own
    conformance test replays, which is how `verify` and `parse` are checked
    against something GitHub actually sent rather than against each other.
    """
    import flyteplugins.github as plugin

    secret = "a-test-signing-secret"
    sign, body = plugin.SAMPLE_DELIVERY
    headers = sign(body, secret)

    assert plugin.verify(body, headers, secret), "a correctly signed delivery must verify"
    assert not plugin.verify(body, headers, "wrong-secret"), "a bad signature must not"

    event = plugin.parse(headers, body)
    return {
        # What `on_event` matches on: `events.PullRequest.OPENED` is this string.
        "qualified_type": event.qualified_type,
        # What `scopes` filters on.
        "scope": event.scope or "",
        "title": event.title or "",
        "actor": event.actor or "",
        # What `run_once` dedupes on.
        "dedupe_key": event.dedupe_key(),
    }
# {{/docs-fragment replay}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(replay_sample_delivery)
    print(run.name)
    print(run.url)
    run.wait()
