# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
# ]
# ///

"""Long-lived agent that persists memory to a keyed blob-store namespace.

Gives a ``flyte.ai.agents.Agent`` continuity across runs by loading a
deterministic ``MemoryStore`` with ``MemoryStore.get_or_create(key=...)``.

The store lives under the stable raw-data root in the Flyte-managed
``agents/memory-store/v0`` namespace. It holds ``messages.json`` (the live
transcript), an opt-in ``audit/log.jsonl`` audit trail, and any path-addressed
artifacts the agent / its tools have written.
"""

from __future__ import annotations

import flyte
from flyte.ai.agents import Agent, ConcurrencyError, MemoryStore

MEMORY_KEY = "my-assistant"
NOTES_PATH = "notes/notes.json"

env = flyte.TaskEnvironment(
    name="persistent-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


# {{docs-fragment tools}}
@env.task
async def add_note(note: str) -> str:
    """Save a free-form note to the agent's scratchpad."""
    memory = await MemoryStore.get_or_create.aio(key=MEMORY_KEY)
    notes = await memory.read_json.aio(NOTES_PATH, default=[])
    sha = await memory.current_sha.aio(NOTES_PATH)
    notes.append(note)
    try:
        # Optimistic concurrency: the write succeeds only if no other writer
        # changed the file between our read and write.
        await memory.write_json.aio(NOTES_PATH, notes, expected_sha=sha, reason="agent note")
    except ConcurrencyError:
        return "Memory changed while saving the note; please retry add_note."
    await memory.save.aio()
    return f"Noted: {note}"


@env.task
async def list_history(count: int = 5) -> str:
    """Return recent persisted notes and conversation messages."""
    memory = await MemoryStore.get_or_create.aio(key=MEMORY_KEY)
    notes = await memory.read_json.aio(NOTES_PATH, default=[])
    return "Persisted notes:\n" + "\n".join(f"- {note}" for note in notes[-count:])
# {{/docs-fragment tools}}


agent = Agent(
    name="memory-assistant",
    instructions=(
        "You are a continuity-aware assistant. You can record notes and look "
        "up recent history. When the user asks you to remember something, call "
        "add_note. When the user asks you to recall something, call list_history."
    ),
    model="claude-haiku-4-5",
    tools=[add_note, list_history],
    max_turns=12,
)


# {{docs-fragment chat}}
@env.task(report=True)
async def chat(message: str, memory_key: str = MEMORY_KEY) -> str:
    """One conversation turn that picks up where the last run left off."""
    # Load (or create) the keyed store; restores the prior transcript + artifacts.
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    flyte.logger.info("Restored %d prior messages from memory.", len(memory.messages))

    # Attach memory to the agent. The prior transcript is prepended to the
    # conversation, and the in-flight transcript is appended back to it.
    agent.memory = memory
    result = await agent.run.aio(message)

    # Persist the updated transcript + any tool-written artifacts back to the key.
    await memory.save.aio()
    return result.summary or result.error
# {{/docs-fragment chat}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(chat, message="Remember that my favorite color is teal and my dog is named Mochi.")
    print(f"First run: {run.url}")
    run.wait()
    print(f"First reply: {run.outputs()[0]}")

    run2 = flyte.run(chat, message="What is my dog's name and favorite color?")
    print(f"Second run: {run2.url}")
    run2.wait()
    print(f"Second reply: {run2.outputs()[0]}")
