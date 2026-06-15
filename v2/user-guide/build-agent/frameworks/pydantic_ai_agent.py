# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "pydantic-ai-slim[anthropic]",
# ]
# main = "run_agent"
# params = 'prompt="What is 17 + 23 plus the temperature in NYC?"'
# ///

"""A PydanticAI agent running inside a Flyte task, with tools backed by durable tasks."""

from __future__ import annotations

# {{docs-fragment all}}
from pydantic_ai import Agent

import flyte

env = flyte.TaskEnvironment(
    name="pydantic-ai-agent",
    image=flyte.Image.from_debian_base(python_version=(3, 13)).with_pip_packages(
        "pydantic-ai-slim[anthropic]",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="ANTHROPIC_API_KEY")],
)


# Durable Flyte tasks that do the real work (heavy IO / compute, retryable).
@env.task
async def fetch_weather(city: str) -> dict[str, float | str]:
    """Fetch a weather snapshot for a city."""
    # Replace with a real weather API call.
    return {"city": city, "temperature_f": 68.4, "conditions": "partly cloudy"}


@env.task
async def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


# The PydanticAI agent. Tools delegate to the durable tasks above.
agent = Agent(
    "anthropic:claude-3-5-sonnet-latest",
    system_prompt=(
        "You are a friendly assistant. Use the tools to look up weather and "
        "compute math. Reply with a single-sentence summary."
    ),
)


@agent.tool_plain
async def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return await fetch_weather(city)


@agent.tool_plain
async def add_numbers(x: float, y: float) -> float:
    """Add two numbers together."""
    return await add(x, y)


@env.task(report=True)
async def run_agent(prompt: str) -> str:
    """Run the PydanticAI agent inside a durable Flyte task."""
    with flyte.group("pydantic-ai-run"):
        result = await agent.run(prompt)
    return result.output
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(run_agent, prompt="What's 17 + 23 plus the temperature in NYC?")
    print(run.url)
