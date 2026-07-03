"""Serve the Code Mode stock analyst with Flyte's native chat app.

``AgentChatAppEnvironment`` provides the whole web layer: the chat UI, the
``/api/chat`` endpoint, progress streaming, and the tools sidebar. Pointing its
``task_entrypoint`` at the ``analyze`` task makes every question a durable Flyte
run, and ``passthrough_auth=True`` forwards the caller's credentials so those
runs launch as the signed-in user (no service identity or org plumbing needed).

The agent pulls live prices from the Yahoo Finance MCP server (no credentials
needed) and runs the DuckDB analytics as a durable task.

Run::

    flyte create secret anthropic_api_key <your-anthropic-key>
    python app.py
"""

import flyte
import flyte.app
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

from analysis import agent, analyze, env as agent_env

_prompt_nudges = [
    {
        "label": "Compare two stocks",
        "prompt": "Compare AAPL and MSFT over the last year — normalized price trend and volatility.",
    },
    {
        "label": "Trend + moving average",
        "prompt": "Show NVDA's closing price with a 50-day moving average for the last year.",
    },
    {
        "label": "Best performer",
        "prompt": "Which of AAPL, MSFT, GOOGL and AMZN had the best 6-month return?",
    },
    {
        "label": "Volatility ranking",
        "prompt": "Rank AAPL, TSLA and NVDA by 3-month volatility.",
    },
]

# {{docs-fragment chat_app}}
env = AgentChatAppEnvironment(
    name="code-mode-analytics",
    agent=agent,  # powers the tools sidebar
    # Each question is launched as a durable run of `analyze` (with the chat
    # history), so the sandbox's query calls dispatch as tracked child tasks.
    task_entrypoint=analyze,
    # Run those tasks with the caller's forwarded credentials.
    passthrough_auth=True,
    title="Code Mode Stock Analytics",
    subtitle=(
        "Chat with live stock prices. The model writes one Python program per "
        "question; it fetches prices from the Yahoo Finance MCP server and runs "
        "the heavy queries as durable Flyte tasks."
    ),
    prompt_nudges=_prompt_nudges,
    theme=CustomTheme(
        accent_color="#0ea5e9",
        accent_hover_color="#0284c7",
        button_text_color="#ffffff",
    ),
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    scaling=flyte.app.Scaling(replicas=1),
    depends_on=[agent_env],
    # Every request launches a run (compute + a paid LLM call), so gate the app
    # behind platform auth.
    requires_auth=True,
)
# {{/docs-fragment chat_app}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    # Remote image builder so no local Docker is needed to build the app + task images.
    flyte.init_from_config(image_builder="remote")

    handle = flyte.serve(env)
    print(f"Deployed Code Mode Stock Analytics: {handle.url}")
# {{/docs-fragment deploy}}
