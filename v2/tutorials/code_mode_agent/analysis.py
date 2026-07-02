"""The durable half: the code-mode analysis task and its tools.

Kept separate from ``app.py`` on purpose. This module is imported and run in the
task environment (which has anthropic / duckdb / monty but *not* fastapi), so it
must not import the web layer. ``app.py`` imports ``analyze`` from here and adds
the FastAPI front end.
"""

from __future__ import annotations

from pydantic import BaseModel

import flyte

import tools
from agent import CodeModeAgent

# {{docs-fragment env}}
ANTHROPIC_SECRET = flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")

# The analysis environment: the agent runs here, and `query` is a durable task in
# the same environment, so the sandboxed code's query calls dispatch as child tasks.
env = flyte.TaskEnvironment(
    name="code-mode",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "anthropic", "pydantic-monty", "duckdb>=1.1.0", "pandas"
    ),
    secrets=[ANTHROPIC_SECRET],
    cache="auto",
)
# {{/docs-fragment env}}


# {{docs-fragment query_task}}
@env.task
async def query(sql: str) -> list:
    """Run a read-only SQL query over the `orders` table and return rows.

    Args:
        sql: A DuckDB SELECT statement against the table `orders`
             (columns: order_date, region, category, channel, revenue, units,
             is_returned). Aggregate in SQL where you can.

    Returns:
        A list of row dicts (one per result row), with dates as ISO strings.
    """
    return await tools.query(sql)
# {{/docs-fragment query_task}}


class ChatResponse(BaseModel):
    code: str = ""
    blocks: list[str] = []
    summary: str = ""
    error: str = ""
    run_url: str = ""
    # Which tools the generated code called, and how many times: [{"name", "count"}].
    tools_used: list[dict] = []


# {{docs-fragment registry}}
# `tools` describe the tools to the model (signatures + docstrings). `execution_tools`
# are what actually run: the durable query task, plus the cheap helpers in-process. The
# two registries share names, so the model writes against one contract and each call is
# routed to the durable or in-process implementation.
agent = CodeModeAgent(
    tools={
        "query": tools.query,
        "create_metric": tools.create_metric,
        "create_chart": tools.create_chart,
        "create_table": tools.create_table,
        "calculate_statistics": tools.calculate_statistics,
    },
    execution_tools={
        "query": query,  # durable @env.task
        "create_metric": tools.create_metric,  # in-process
        "create_chart": tools.create_chart,  # in-process
        "create_table": tools.create_table,  # in-process
        "calculate_statistics": tools.calculate_statistics,  # in-process
    },
    context=tools.DATASET_DESCRIPTION,
    model="claude-opus-4-8",
    max_retries=2,
)
# {{/docs-fragment registry}}


# {{docs-fragment analyze}}
@env.task
async def analyze(message: str) -> ChatResponse:
    """Run one code-mode analysis: generate code, run it in the sandbox, return results.

    Runs inside a task context, so the sandbox's `query` calls dispatch as durable
    child tasks.
    """
    result = await agent.run(message, [])
    return ChatResponse(
        code=result.code,
        blocks=result.blocks,
        summary=result.summary,
        error=result.error,
        tools_used=result.tools_used,
    )
# {{/docs-fragment analyze}}


def tool_descriptions() -> list[dict]:
    """Tool metadata for the UI sidebar (auto-generated from the registry)."""
    return agent.tool_descriptions()


if __name__ == "__main__":
    # Run one analysis as a durable flyte.run (no app) — handy for testing the
    # analysis half on its own. Remote image builder so no local Docker is needed.
    flyte.init_from_config(image_builder="remote")
    run = flyte.run(analyze, message="Show me monthly revenue for 2024")
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs()[0])
