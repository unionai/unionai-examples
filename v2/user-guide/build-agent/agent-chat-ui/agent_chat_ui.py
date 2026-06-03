# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte>=2.0.0",
#     "litellm",
#     "fastapi",
#     "uvicorn",
#     "httpx",
# ]
# ///

"""Run ``Agent`` behind the built-in chat UI.

Because ``flyte.ai.agents.Agent`` implements the ``AgentProtocol``, it plugs
straight into ``flyte.ai.chat.AgentChatAppEnvironment`` — you get a hosted chat
shell, tool sidebar, and NDJSON streaming for free.
"""

from __future__ import annotations

import pathlib

# {{docs-fragment all}}
import flyte
from flyte.ai.agents import Agent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

task_env = flyte.TaskEnvironment(
    name="chat-agent-tools",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm", "httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@task_env.task
async def search_docs(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search internal documentation (stub) and return matching snippets."""
    corpus = [
        {"title": "Tasks", "body": "Define a task by decorating an async function with @env.task."},
        {"title": "Triggers", "body": "Schedule a task by attaching a flyte.Trigger with a flyte.Cron automation."},
        {"title": "Secrets", "body": "Mount cluster-managed secrets into a task with flyte.Secret(...)."},
    ]
    needle = query.lower()
    matches = [d for d in corpus if needle in d["body"].lower() or needle in d["title"].lower()]
    return matches[:max_results]


agent = Agent(
    name="docs-helper",
    instructions=(
        "You are a friendly internal docs assistant. Use search_docs to find "
        "relevant snippets. Always cite the doc title in your final answer."
    ),
    model="claude-haiku-4-5",
    tools=[search_docs],
    max_turns=8,
)


@task_env.task(report=True)
async def chat_entrypoint(message: str, history: list[dict]) -> dict:
    """Parent task that owns the agent loop and the nested tool tasks."""
    result = await agent.run.aio(message, history=history)
    return {
        "summary": result.summary,
        "error": result.error,
        "attempts": result.attempts,
        "charts": [],
        "code": "",
    }


env = AgentChatAppEnvironment(
    name="docs-agent-chat-ui",
    agent=agent,
    task_entrypoint=chat_entrypoint,
    title="Internal docs assistant",
    subtitle="Backed by a flyte.ai.agents.Agent + durable Flyte task tools.",
    theme=CustomTheme(accent_color="#6F2AEF", accent_hover_color="#8B52F2"),
    prompt_nudges=[
        {"label": "Basics", "prompt": "Can you show me a hello world example?"},
        {"label": "Triggers", "prompt": "How do I schedule a task?"},
    ],
    depends_on=[task_env],
    image=flyte.Image.from_debian_base().with_pip_packages("litellm", "fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env)
    print(f"Agent chat UI deployed: {deployments[0].summary_repr()}")
    print(f"Url: {deployments[0].envs['docs-agent-chat-ui'].deployed_app.url}")
