"""The durable half: the code-mode analysis task, built on ``flyte.ai.agents``.

The agent is Flyte's native :class:`~flyte.ai.agents.Agent` with ``code_mode=True``:
on each turn the model writes a small Python program, the program runs in the Monty
sandbox, and the tools are exposed to it as plain functions. The Yahoo Finance MCP
server supplies the live price fetch; the durable ``query`` task runs the DuckDB
analytics on the cluster; the render helpers run in-process and stream their HTML
into the report collector in ``tools.py``.

Kept separate from ``app.py`` on purpose. This module runs in the task image (which
has anthropic / duckdb / monty / the MCP client but not the web layer), and
``app.py`` serves it.
"""

from __future__ import annotations

from typing import Any

import flyte
import flyte.remote
from flyte.ai.agents import Agent, LLMMessage, MCPServerSpec
from flyte.ai.agents._code import build_sandbox_tools, extract_python_code
from flyte.ai.agents._tools import _abbreviate
from flyte.ai.agents.agent import AgentEvent, _TurnResult, _emit
from flyte.ai.agents.protocol import AgentResult

import tools

# {{docs-fragment env}}
ANTHROPIC_SECRET = flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")

# The analysis environment: the agent runs here, and `query` is a durable task in
# the same environment, so the sandboxed code's query calls dispatch as child tasks.
# The Yahoo Finance MCP server needs no credentials (public data), so the only secret
# is the Anthropic key.
env = flyte.TaskEnvironment(
    name="code-mode",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "flyte[mcp]",
        "anthropic",
        "pydantic-monty",
        "duckdb>=1.1.0",
        "pandas",
        "mcp-yahoo-finance",
    ),
    secrets=[ANTHROPIC_SECRET],
)
# {{/docs-fragment env}}


# {{docs-fragment query_task}}
# Cache the analytics: given the same SQL over the same fetched series, the result is
# deterministic, so identical queries dedupe across conversations. The fetch itself is
# live and is not cached — it is an MCP tool call the agent makes, not this task.
@env.task(cache="auto")
async def _query_task(sql: str, series: dict[str, str]) -> list[dict]:
    return await tools.run_sql(sql, series)


async def query(sql: str, series: dict[str, str]) -> list[dict]:
    """Run a read-only SQL query over fetched stock prices and return rows.

    Args:
        sql: A DuckDB SELECT statement against the table `prices`
             (columns: ticker, date, close). Aggregate in SQL where you can.
        series: Maps ticker symbol -> the JSON string returned by
                yf_get_historical_stock_prices for it. Pass the raw strings; the
                durable task parses them into the `prices` table.

    Returns:
        A list of row dicts (one per result row), with dates as ISO strings.
    """

    return await _query_task(sql, series)


# {{/docs-fragment query_task}}


# {{docs-fragment llm}}
async def call_llm(
    model: str, system: str, messages: list[dict], tools_schema: list[dict] | None
) -> LLMMessage:
    """LLM callback for the agent, using the official Anthropic SDK.

    The agent's default callback goes through litellm; supplying our own keeps the
    image lean and the API surface explicit. In code mode `tools_schema` is None
    (tools are called from generated code, not via JSON tool-calling).
    """

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY, injected as a Flyte secret
    resp = await client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=messages,
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return LLMMessage(content=text)


# {{/docs-fragment llm}}


# {{docs-fragment mcp}}
def _mcp_servers() -> list[MCPServerSpec]:
    """The Yahoo Finance MCP server — the agent's live price source.

    The agent connects on first use, lists the server's tools, and registers each
    one alongside the local tools; the model calls them from its generated code
    like any other function. `tool_prefix` namespaces them, and `tool_filter`
    narrows the server's 12 tools down to the one the analytics needs, the same
    surface-shrinking move as the SQL guard. No auth: the server reads public
    Yahoo Finance data, so there is no secret to inject.
    """
    return [
        MCPServerSpec(
            name="yahoo-finance",
            command=["mcp-yahoo-finance"],
            tool_prefix="yf_",
            tool_filter=["get_historical_stock_prices"],
        )
    ]


# {{/docs-fragment mcp}}


INSTRUCTIONS = f"""\
You are a stock-market data analyst in a chat. Answer questions by writing one
complete Python program that fetches the price history you need and assembles a report.

{tools.DATA_DESCRIPTION}

How to build the report:
- IMPORTANT: the user only sees what you RENDER. The value your code returns and the
  rows from query(...) are NOT shown to them. You must turn your findings into report
  blocks with create_metric / create_chart / create_table, or the user sees nothing. A
  reply that describes a chart without calling create_chart shows an empty answer.
- Fetch each ticker the question needs with yf_get_historical_stock_prices(...). When
  you need more than one, call it once per ticker (await each call). Pass the raw JSON
  strings straight through — do not parse them (the sandbox has no json/datetime).
- Do all the analytics in SQL via query(sql, series): build series as
  {{"AAPL": aapl_json, "MSFT": msft_json}} and write one SELECT against the `prices`
  table (columns ticker, date, close). Use window functions, LAG, and STDDEV — do not
  compute these by hand in Python.
- Build a report, not just one chart. Lead with one or two headline numbers via
  create_metric(...), then a create_chart(...) for the trend, and a create_table(...)
  when the exact figures matter. Use the tools that fit the question.
- The create_* tools add blocks to the report in the order you call them; they return
  short confirmations, not HTML.
- Fetch each ticker once and run each query once; reuse the returned rows for every
  metric, chart, and table.
- After one successful code block has created the report, stop writing code and give
  the final plain-text summary. Do not re-run the analysis in a second code block
  unless the prior code failed.
- Format numbers with f-strings, e.g. f"${{x:.2f}}" or f"{{r:.1%}}". The format() builtin
  and the {{:,}} thousands separator are not available in the sandbox.
- Prefer ONE code block that does everything: fetch, query, render. After it runs,
  reply with a one-or-two-sentence plain-text summary of what the data shows.
- Your final reply is rendered as Markdown. Write "about 12%", never "~12%": a pair of
  ~ characters renders as strikethrough.
- For a greeting or a question that needs no data, just reply in plain text.
"""


class CodeModeAgent(Agent):
    """Agent shim for flyte 2.5.7 code-mode edge cases used by this tutorial."""

    async def _run_code_mode(
        self,
        message: str,
        memory: Any = None,
    ) -> AgentResult:
        import flyte.sandbox

        # flyte 2.5.7 loads MCP inside _run_loop, after code mode has already
        # snapshotted sandbox_tools. Load first so the yf_* MCP tools enter Monty's
        # namespace and the generated code can call them.
        await self._ensure_mcp_loaded()
        sandbox_tools = build_sandbox_tools(
            self._registry, call_llm=self.call_llm, model=self.model
        )
        last_code = ""
        sandbox_runs = 0
        report_created = False
        render_nudged = False

        async def step(
            llm_msg: LLMMessage, messages: list[dict[str, Any]], attempts: int
        ) -> _TurnResult:
            nonlocal last_code, sandbox_runs, report_created, render_nudged
            text = llm_msg.content or ""
            messages.append({"role": "assistant", "content": text})
            await _emit(AgentEvent("message", {"role": "assistant", "content": text}))

            code = extract_python_code(text)

            # Once the report exists, the model's next message is its plain-text
            # summary. Ignore any further code — the report is done and we do not
            # want a second run re-executing the same queries.
            if report_created:
                summary = text if not code else "Done. The report is above."
                await _emit(AgentEvent("turn_end", {"turn": attempts, "summary": True}))
                return _TurnResult(done=True, final_text=summary)

            if not code:
                # No report and no code: a greeting or a question needing no data.
                await _emit(
                    AgentEvent(
                        "turn_end",
                        {"turn": attempts, "had_code": False, "text_len": len(text)},
                    )
                )
                return _TurnResult(done=True, final_text=text)

            last_code = code
            sandbox_runs += 1
            await _emit(AgentEvent("tool_start", {"tool": "<sandbox>", "code": code}))
            try:
                with flyte.group(f"{self.name}-sandbox-{sandbox_runs}"):
                    result = await flyte.sandbox.orchestrate_local(
                        code,
                        inputs={"_unused": 0},
                        tasks=sandbox_tools,
                    )
            except Exception as exc:
                await _emit(
                    AgentEvent("tool_error", {"tool": "<sandbox>", "error": str(exc)})
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your code raised an error:\n\n```\n{exc}\n```\n\n"
                            "Fix the code and try again, respecting the Monty sandbox restrictions."
                        ),
                    }
                )
                await _emit(
                    AgentEvent(
                        "turn_end", {"turn": attempts, "had_code": True, "error": True}
                    )
                )
                return _TurnResult(done=False)

            await _emit(
                AgentEvent(
                    "tool_end", {"tool": "<sandbox>", "result": _abbreviate(result)}
                )
            )
            await _emit(
                AgentEvent(
                    "turn_end",
                    {"turn": attempts, "had_code": True, "final_after_code": True},
                )
            )
            # Only treat the turn as done when the render tools actually produced
            # report blocks. If the model computed a result but rendered nothing,
            # the user would see an empty answer (the query rows are not shown), and
            # asking for a summary here would make the model narrate a report that
            # does not exist. So check the collector and nudge it to render first.
            blocks = tools.collect_report()
            if blocks:
                report_created = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The report has been created and is shown to the user. "
                            "Reply with a one or two sentence plain-text summary of "
                            "what the data shows. Do not write any more code."
                        ),
                    }
                )
                return _TurnResult(done=False)
            if not render_nudged:
                render_nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your code ran but added nothing to the report, so the "
                            "user sees no result — the query rows are not displayed "
                            "automatically. Call create_metric / create_chart / "
                            "create_table to render the findings, then stop."
                        ),
                    }
                )
                return _TurnResult(done=False)
            # Rendered nothing even after a nudge: end honestly rather than claim a
            # report that was never built.
            return _TurnResult(
                done=True,
                final_text="I ran the analysis but did not produce a visual report.",
            )

        outcome = await self._run_loop(
            message, memory, tools_schema=None, step=step, mode="code"
        )
        return AgentResult(
            code=last_code,
            summary=outcome.last_text,
            error=outcome.error_msg,
            attempts=outcome.attempts,
            memory=outcome.memory,
        )


# {{docs-fragment agent}}
agent = CodeModeAgent(
    name="code-mode-analyst",
    instructions=INSTRUCTIONS,
    model="claude-opus-4-8",
    # One list, two kinds of local tools: `query` awaits an @env.task, so the sandbox
    # dispatches the DuckDB analytics as a durable child task; the render helpers are
    # plain callables and run in-process. The live price fetch is a *third* kind — an
    # MCP tool contributed by `mcp_servers` below — but the model calls all of them the
    # same way. The agent introspects signatures and docstrings to build its prompt.
    tools=[
        query,
        tools.create_metric,
        tools.create_chart,
        tools.create_table,
        tools.calculate_statistics,
    ],
    mcp_servers=_mcp_servers(),
    code_mode=True,
    # Turn 1 writes the program; the next turn is the plain-text summary. The
    # spare turns let the agent fix its code if the sandbox rejects it.
    max_turns=5,
    call_llm=call_llm,
)
# {{/docs-fragment agent}}

# Shows up in the task logs, so a deployment is easy to spot as MCP-enabled.
print(f"Yahoo Finance MCP: {len(agent.mcp_servers)} server(s) configured")


async def _run_link_block() -> str:
    """A small HTML block linking to this run in the UI (best effort)."""
    tctx = flyte.ctx()
    if tctx is None or not tctx.action.run_name:
        return ""
    try:
        run = await flyte.remote.Run.get.aio(tctx.action.run_name)
        url = run.url
    except Exception:
        return ""
    # Inline light-sky color so the link stays readable on the chat UI's dark theme
    # (an unstyled anchor inherits a dark blue that disappears on the near-black page).
    return (
        '<div class="block" style="margin:10px 0;font-size:13px;">'
        f'<a href="{url}" target="_blank" rel="noopener" '
        'style="color:#7dd3fc;text-decoration:underline;">'
        "View this analysis run in the Union UI &#8599;</a></div>"
    )


# {{docs-fragment analyze}}
@env.task
async def analyze(message: str, history: list[dict[str, str]]) -> dict:
    """Run one analysis: start a report, run the agent, return blocks + summary.

    `history` is the prior conversation, which `Agent.run` takes as its memory, so
    follow-ups can refer back to earlier turns. This task is the chat app's
    `task_entrypoint`: each question becomes a run, and inside it the sandbox's
    `query` calls dispatch as durable child tasks.
    """

    tools.start_report()
    result = await agent.run.aio(message, memory=list(history))
    blocks = tools.collect_report()
    if link := await _run_link_block():
        blocks.append(link)
    # The UI renders the summary as Markdown, where a pair of ~ characters becomes
    # strikethrough. Models like ~ as shorthand for "approximately", so escape it.
    summary = result.summary.replace("~", "\\~")
    return {
        "summary": summary,
        "charts": blocks,
        "code": result.code,
        "error": result.error,
        "attempts": result.attempts,
    }


# {{/docs-fragment analyze}}


if __name__ == "__main__":
    # Run one analysis as a durable flyte.run (no app) — handy for testing the
    # analysis half on its own. Remote image builder so no local Docker is needed.
    flyte.init_from_config(image_builder="remote")
    run = flyte.run(
        analyze, message="Compare AAPL and MSFT over the last 6 months", history=[]
    )
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs()[0])
