# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
# ]
# ///

"""Long-lived research assistant that persists memory to a keyed namespace.

Gives a ``flyte.ai.agents.Agent`` continuity across runs by passing a
deterministic ``MemoryStore`` (loaded with ``MemoryStore.get_or_create(key=...)``)
into every ``agent.run`` call.

The value of ``MemoryStore`` shows up without any bookkeeping tools. Continuity
is automatic because the prior conversation transcript is reloaded and prepended
on each run, so the agent remembers two things for free:

- **What the user told it.** Share a fact in one run, ask about it in the next,
  and it recalls it.
- **What its tools returned.** A ``web_search`` result lands in the transcript,
  so the agent can cite or build on it later without searching again.

The store lives under the stable raw-data root in the Flyte-managed
``agents/memory-store/v0`` namespace. It holds ``messages.json`` (the live
transcript), an opt-in ``audit/log.jsonl`` audit trail, and any path-addressed
artifacts you choose to write.
"""

from __future__ import annotations

import flyte
from flyte.ai.agents import Agent, MemoryStore

MEMORY_KEY = "my-assistant"

env = flyte.TaskEnvironment(
    name="persistent-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


# {{docs-fragment agent}}
@env.task
async def web_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search the web for `query` and return the top matching results.

    A stateless tool — it knows nothing about the agent's memory. But because the
    results it returns are recorded in the conversation transcript, the agent can
    recall or build on them in a later run without searching again.

    This stub returns canned results so the example runs offline. In a real
    agent, replace it with a call to a search API (Tavily, Brave, SerpAPI, …);
    keeping it an `@env.task` makes each search durable, retryable, and
    observable in the dashboard.
    """
    return [
        {
            "title": f"{query.title()} — overview ({i + 1})",
            "url": f"https://example.com/?q={query.replace(' ', '+')}&r={i + 1}",
            "snippet": f"Key point #{i + 1} about {query}.",
        }
        for i in range(max_results)
    ]


agent = Agent(
    name="memory-assistant",
    instructions=(
        "You are a personal research assistant with long-term memory. You "
        "remember what the user is working on and the facts they share, because "
        "your prior conversation transcript is always available. Use web_search "
        "to look things up, and reuse earlier findings from the conversation "
        "instead of searching again when you already have the answer."
    ),
    model="claude-haiku-4-5",
    tools=[web_search],
    max_turns=12,
)
# {{/docs-fragment agent}}


# {{docs-fragment chat}}
@env.task(report=True)
async def chat(message: str, memory_key: str = MEMORY_KEY) -> str:
    """One conversation turn that picks up where the last run left off."""
    # Load (or create) the keyed store; restores the prior transcript.
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    flyte.logger.info("Restored %d prior messages from memory.", len(memory.messages))

    # Memory is passed in per call (not attached to the agent). The prior
    # transcript is prepended to the conversation and this turn is appended back
    # onto the store, which is also returned on result.memory.
    result = await agent.run.aio(message, memory=memory)

    # Saving is explicit — run never persists on its own. Write the updated
    # transcript back to the deterministic keyed remote path.
    await memory.save.aio()
    return result.summary or result.error
# {{/docs-fragment chat}}


if __name__ == "__main__":
    flyte.init_from_config()
    print("First turn — share context and run a search...")
    run = flyte.run(
        chat,
        message="I'm learning to bake sourdough. Search for a few beginner tips and summarize them.",
    )
    print(f"First run: {run.url}")
    run.wait()
    print(f"First reply: {run.outputs()[0]}")

    print("\nSecond turn — the agent recalls the context and findings from memory...")
    run2 = flyte.run(
        chat,
        message="Remind me what I'm learning and what tips you found earlier.",
    )
    print(f"Second run: {run2.url}")
    run2.wait()
    print(f"Second reply: {run2.outputs()[0]}")
