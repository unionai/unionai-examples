"""OpenAI Agents with Flyte, basic tool example.

Usage:

Create secret:

```
flyte create secret openai_api_key
uv run agents_tools.py
```
"""
# {{docs-fragment uv-script}}

# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-openai>=2.0.0b7",
#    "openai-agents>=0.2.4",
#    "pydantic>=2.10.6",
# ]
# main = "main"
# params = ""
# ///

# {{/docs-fragment uv-script}}

# {{docs-fragment imports-task-env}}
from agents import Agent, Runner
from pydantic import BaseModel

import flyte
from flyteplugins.openai.agents import function_tool


env = flyte.TaskEnvironment(
    name="openai_agents_tools",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(__file__, name="openai_agents_image"),
    secrets=flyte.Secret("openai_api_key", as_env_var="OPENAI_API_KEY"),
)

# {{/docs-fragment imports-task-env}}

# {{docs-fragment tools}}
class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


@function_tool
@env.task
async def get_weather(city: str) -> Weather:
    """Get the weather for a given city."""
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")

# {{/docs-fragment tools}}

# {{docs-fragment agent}}
agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


@env.task
async def main() -> str:
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    return result.final_output

# {{/docs-fragment agent}}

# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
# {{/docs-fragment main}}
