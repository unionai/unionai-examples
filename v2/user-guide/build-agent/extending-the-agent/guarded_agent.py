# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
# ]
# ///

"""Extend the Agent class by overriding ``run`` to add input/output guardrails."""

from __future__ import annotations

# {{docs-fragment guarded}}
from dataclasses import dataclass

from flyte.ai.agents import Agent
from flyte.ai.agents.protocol import AgentResult
from flyte.syncify import syncify


@dataclass(kw_only=True)
class GuardedAgent(Agent):
    """An Agent that rejects banned input and signs its final answer."""

    banned_terms: tuple[str, ...] = ()
    signature: str = ""

    @syncify
    async def run(self, message: str, history: list[dict] | None = None) -> AgentResult:
        # 1. Pre-flight input guardrail — short-circuit without an LLM call.
        lowered = message.lower()
        if any(term in lowered for term in self.banned_terms):
            return AgentResult(error="Request rejected by input guardrail.")

        # 2. Delegate to the built-in tool-use loop.
        result = await super().run.aio(message, history)

        # 3. Post-process the final answer.
        if result.summary and self.signature:
            result.summary = f"{result.summary.strip()}\n\n— {self.signature}"
        return result
# {{/docs-fragment guarded}}


async def echo(text: str) -> str:
    """Echo the input text back."""
    return text


agent = GuardedAgent(
    name="guarded-helper",
    instructions="You are a careful assistant.",
    tools=[echo],
    banned_terms=("ssn", "password"),
    signature="GuardedAgent",
)


if __name__ == "__main__":
    result = agent.run("Summarize today's open tickets.")
    print(result.summary or result.error)
