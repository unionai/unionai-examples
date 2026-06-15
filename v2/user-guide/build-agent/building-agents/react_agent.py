# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "openai",
#     "pydantic",
# ]
# main = "react_agent"
# params = 'goal="What is (12 + 8) * 3?"'
# ///

# {{docs-fragment all}}
import json

from openai import AsyncOpenAI
from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(
    name="agent_env",
    image=flyte.Image.from_debian_base(python_version=(3, 13)).with_pip_packages("openai"),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[flyte.Secret(key="OPENAI_API_KEY")],
)

TOOLS = {"add": lambda a, b: a + b, "multiply": lambda a, b: a * b}


@flyte.trace  # each call = a span in the dashboard
async def reason(goal: str, history: str) -> dict:
    """LLM picks a tool or returns a final answer."""
    r = await AsyncOpenAI().chat.completions.create(
        model="gpt-4.1-nano",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"Tools: {list(TOOLS)}. Respond JSON: "
                '{"thought":..,"tool":..,"args":{}} or {"thought":..,"done":true,"answer":..}',
            },
            {"role": "user", "content": f"Goal: {goal}\n\n{history}\nWhat next?"},
        ],
    )
    return json.loads(r.choices[0].message.content)


@flyte.trace
async def act(tool: str, args: dict) -> str:
    """Execute the chosen tool."""
    return str(TOOLS[tool](**args))


class AgentResult(BaseModel):
    answer: str
    steps: int


@env.task  # runs in its own container
async def react_agent(goal: str, max_steps: int = 10) -> AgentResult:
    history = ""
    for step in range(1, max_steps + 1):  # the agent loop
        decision = await reason(goal, history)  # Thought
        if decision.get("done"):
            return AgentResult(answer=str(decision["answer"]), steps=step)
        result = await act(decision["tool"], decision["args"])  # Action
        # Observation
        history += f"Step {step}: {decision['thought']} -> {decision['tool']}({decision['args']}) = {result}\n"
    return AgentResult(answer="Max steps reached", steps=max_steps)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(react_agent, goal="What is (12 + 8) * 3?")
    print(run.url)
    run.wait()
    print(run.outputs())
