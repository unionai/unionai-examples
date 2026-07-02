"""CodeModeAgent — an LLM writes code, a sandbox runs it.

Instead of calling tools one at a time, the model writes a single Python program
that orchestrates the tools, and that program runs in Flyte's Monty sandbox. The
sandbox allows no imports, no IO, and no network, so the generated code can only
call the tools you register. The tools do the real work; in ``app.py`` the heavy
one (``query``) is a durable Flyte task.

The loop:

    build prompt from the tool registry -> ask Claude for code -> run it in the
    sandbox -> on failure, hand the error back to Claude to fix -> repeat.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import flyte
import flyte.sandbox

# ---------------------------------------------------------------------------
# LLM call + code extraction (module-level so @flyte.trace can wrap them)
# ---------------------------------------------------------------------------


async def _call_llm(model: str, system: str, messages: list[dict[str, str]]) -> str:
    """Send a chat-completion request to Claude and return the text response."""
    from anthropic import AsyncAnthropic

    # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment (injected as a
    # Flyte secret in app.py).
    client = AsyncAnthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _extract_code(text: str) -> str:
    """Pull Python code out of a markdown fence, or return the raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


@flyte.trace
async def generate_code(model: str, system: str, messages: list[dict[str, str]]) -> str:
    """Call Claude to generate analysis code and extract it (traced for observability)."""
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


@dataclass
class AgentResult:
    """Outcome of a single ``CodeModeAgent.run`` invocation."""

    code: str = ""
    blocks: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    attempts: int = 1
    tools_used: list[dict] = field(default_factory=list)


def _tool_calls(code: str, tool_names: list[str]) -> list[dict]:
    """Count how many times each registered tool is called in the generated code."""
    used = []
    for name in tool_names:
        n = len(re.findall(r"\b" + re.escape(name) + r"\s*\(", code))
        if n:
            used.append({"name": name, "count": n})
    return used


class CodeModeAgent:
    """Generate analysis code with an LLM, run it in a Monty sandbox, retry on failure.

    Parameters
    ----------
    tools:
        Mapping of tool-name -> callable used to *describe* the tools in the
        system prompt (signatures + docstrings are introspected).
    execution_tools:
        Mapping of tool-name -> callable used at *execution* time in the sandbox.
        Pass ``@env.task``-wrapped callables here for durable execution; plain
        functions run in-process. Defaults to ``tools``.
    context:
        Extra text appended to the system prompt (e.g. the dataset schema).
    model:
        Anthropic model id.
    max_retries:
        Additional attempts after the first failure.
    """

    def __init__(
        self,
        tools: dict[str, Callable],
        *,
        execution_tools: dict[str, Callable] | None = None,
        context: str = "",
        model: str = "claude-opus-4-8",
        max_retries: int = 2,
    ) -> None:
        self._tools = tools
        self._execution_tools = execution_tools or tools
        self._context = context
        self._model = model
        self._max_retries = max_retries
        self.system_prompt = self._build_system_prompt()

    # -- Prompt generation -------------------------------------------------

    def _build_system_prompt(self) -> str:
        # {{docs-fragment describe_tools}}
        # Turn the tool registry into the prompt: introspect each function's signature
        # and docstring, so adding a tool needs no prompt edits.
        tool_lines: list[str] = []
        for name, fn in self._tools.items():
            sig = inspect.signature(fn)
            doc = textwrap.indent(inspect.getdoc(fn) or "", "        ")
            tool_lines.append(f"    - {name}{sig}\n{doc}")
        tools_block = "\n\n".join(tool_lines)
        # {{/docs-fragment describe_tools}}

        # ORCHESTRATOR_SYNTAX_PROMPT uses literal braces; escape them for .format-free
        # string building (we use .replace below, so this is just for safety).
        restrictions = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT

        return (
            textwrap.dedent("""\
            You are a data analyst. Write one Python program that analyzes the data
            and assembles a small report. The program runs in a restricted sandbox.

            {context}

            Available functions:
        {tools}

            {restrictions}
            - Do the heavy lifting in SQL via query(...); shape results with list
              comprehensions (no dict mutation), then render them.
            - Build a report, not just one chart. Lead with one or two headline numbers
              via create_metric(...), then a create_chart(...) for the trend or breakdown,
              and a create_table(...) when the exact figures matter. Use the tools that fit
              the question; you do not have to use all of them.
            - Format numbers with f-strings, e.g. f"${x/1000000:.2f}M" or f"{rate:.1%}". The
              format() builtin and the {:,} thousands separator are not available here.
            - Return a dict: {"blocks": [<html strings, in display order>], "summary": "<one or two sentences>"}
              Every create_* function returns an HTML string. Put them in "blocks" in the
              order you want them shown.

            Example — a revenue overview:
                rows = query(
                    "SELECT monthname(order_date) AS month, region, SUM(revenue) AS revenue "
                    "FROM orders GROUP BY month, region"
                )
                months = ["January","February","March","April","May","June",
                          "July","August","September","October","November","December"]
                regions = ["North","South","East","West"]
                series = []
                for region in regions:
                    by_month = {r["month"]: r["revenue"] for r in rows if r["region"] == region}
                    series.append({"label": region, "data": [by_month.get(m, 0) for m in months]})
                total = sum(r["revenue"] for r in rows)
                by_region = query(
                    "SELECT region, SUM(revenue) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC"
                )
                metric = create_metric("Total 2024 revenue", f"${total/1000000:.2f}M")
                chart = create_chart("line", "Revenue by Region", months, series)
                table = create_table("Revenue by region", ["Region", "Revenue"],
                                     [[r["region"], f"${r['revenue']/1000:.0f}K"] for r in by_region])
                {"blocks": [metric, chart, table],
                 "summary": "Revenue climbs into Q4, led by the " + by_region[0]["region"] + " region."}
        """)
            .replace("{context}", self._context)
            .replace("{tools}", tools_block)
            .replace("{restrictions}", restrictions)
        )

    def tool_descriptions(self) -> list[dict]:
        """JSON-friendly metadata for every registered tool (for the UI sidebar).

        ``durable`` is True when the tool's execution binding was wrapped as a task
        (so its call dispatches as a child task) rather than run in-process.
        """
        out: list[dict] = []
        for name, fn in self._tools.items():
            doc = inspect.getdoc(fn) or ""
            durable = self._execution_tools.get(name) is not fn
            out.append(
                {
                    "name": name,
                    "signature": f"{name}{inspect.signature(fn)}",
                    "description": doc.split("\n\n")[0].replace("\n", " ").strip(),
                    "durable": durable,
                }
            )
        return out

    # -- Sandbox execution -------------------------------------------------

    # {{docs-fragment execute}}
    async def _execute(self, code: str) -> Any:
        """Run *code* in a Monty sandbox with the registered execution tools."""
        # orchestrate_local classifies each execution tool: a TaskTemplate is dispatched
        # as a durable child task, a plain callable runs in-process. So `query` (a task)
        # and the render helpers (plain functions) sit side by side in one sandbox call.
        # Monty requires at least one input; pass a dummy since the generated code
        # takes none.
        return await flyte.sandbox.orchestrate_local(
            code,
            inputs={"_unused": 0},
            tasks=list(self._execution_tools.values()),
        )
    # {{/docs-fragment execute}}

    # -- Main entry point --------------------------------------------------

    # {{docs-fragment run}}
    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        """Generate code, run it in the sandbox, and retry on failure."""
        messages: list[dict[str, str]] = [*history, {"role": "user", "content": message}]

        try:
            code = await generate_code(self._model, self.system_prompt, messages)
        except Exception as exc:
            return AgentResult(error=f"Code generation failed: {exc}")

        attempts = 1
        for attempt in range(1 + self._max_retries):
            attempts = attempt + 1
            try:
                result = await self._execute(code)
            except Exception as exc:
                if attempt < self._max_retries:
                    retry = (
                        f"Your previous code failed with this error:\n\n```\n{exc}\n```\n\n"
                        f"The code that failed:\n\n```python\n{code}\n```\n\n"
                        "Please fix it. Remember the sandbox restrictions."
                    )
                    retry_messages = [
                        *messages,
                        {"role": "assistant", "content": f"```python\n{code}\n```"},
                        {"role": "user", "content": retry},
                    ]
                    try:
                        code = await generate_code(self._model, self.system_prompt, retry_messages)
                    except Exception as llm_exc:
                        return AgentResult(code=code, error=f"Retry LLM call failed: {llm_exc}", attempts=attempts)
                    continue
                return AgentResult(
                    code=code,
                    error=f"Sandbox execution failed after {attempts} attempt(s): {exc}",
                    attempts=attempts,
                )

            blocks = result.get("blocks", []) if isinstance(result, dict) else []
            summary = result.get("summary", "No summary generated.") if isinstance(result, dict) else str(result)
            return AgentResult(
                code=code,
                blocks=blocks,
                summary=summary,
                attempts=attempts,
                tools_used=_tool_calls(code, list(self._tools)),
            )

        return AgentResult(code=code, error="Unexpected: exhausted retries", attempts=attempts)
    # {{/docs-fragment run}}
