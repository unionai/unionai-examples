# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "langgraph==1.0.7",
#     "langchain==1.2.7",
#     "langchain-anthropic==1.3.1",
# ]
# main = "run_agent"
# params = 'query="What is the weather in SF?"'
# ///

"""A single LangGraph agent running inside a Flyte task."""

from __future__ import annotations

# {{docs-fragment all}}
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

import flyte

env = flyte.TaskEnvironment(
    name="langgraph-agent",
    image=flyte.Image.from_debian_base(python_version=(3, 13)).with_pip_packages(
        "langgraph",
        "langchain",
        "langchain-anthropic",
    ),
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="ANTHROPIC_API_KEY")],
)


@env.task
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"


@env.task
async def run_agent(query: str) -> list[BaseMessage]:
    agent = create_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        prompt="You are a helpful assistant.",
    )
    output = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return output["messages"]
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(run_agent, query="What is the weather in SF?")
    print(r.url)
