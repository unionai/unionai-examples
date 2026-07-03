# Code Mode: an LLM writes code, a sandbox runs it, tasks do the work

Chat with live stock prices. You ask a question in the browser, the app launches a
Flyte **run** to answer it, and inside that run Claude writes a small Python program
that executes in Flyte's **Monty sandbox**. The sandbox allows no imports, no IO, and
no network, so the model's code can only call the tools you register: it **fetches**
daily prices from a Yahoo Finance **MCP server** and hands them to a DuckDB **`query`**
that runs as a **durable Flyte task** — a tracked, retryable child task you can click
into.

The whole thing is built on Flyte's native AI stack:

- `flyte.ai.agents.Agent` with `code_mode=True` runs the write-code / run-it /
  self-correct loop.
- `flyte.ai.chat.AgentChatAppEnvironment` provides the chat UI, streaming progress,
  and the `/api/chat` endpoint; its `task_entrypoint` launches one durable run per
  question with the caller's forwarded credentials (`passthrough_auth`).
- `MCPServerSpec` plugs in external tools over MCP; the example wires the
  `mcp-yahoo-finance` server as the agent's live price source (public data, no
  credentials).

## Why "code mode"

Most tool-using agents call tools one at a time: the model asks for a tool, the
result comes back, it reasons, it asks for the next one. For anything multi-step that
is a lot of round-trips. In **code mode** the model instead writes a single program
that orchestrates the tools, with real control flow and composition — here, fetch a
few tickers, hand them to one query, render the result.

Running an LLM's code is normally the scary part. Here it is the safe part: the
program runs in Monty, a restricted interpreter with **no imports, no filesystem, no
network, microsecond startup**. The only things it can do are call the tools you
handed the sandbox.

## What runs where

| Piece | Where it runs | Why |
|---|---|---|
| the chat app | a CPU app pod | Native `AgentChatAppEnvironment`: UI, streaming, one run per question. |
| `analyze` | a Flyte task (the run) | Starts a report, runs the agent loop, returns blocks + summary. |
| `yf_get_historical_stock_prices` | the MCP server subprocess | Live price fetch — the agent's only path to the network. Loaded via `MCPServerSpec`, callable from generated code like any other function. |
| `query(sql, series)` | a **durable child task** | Parses the raw MCP price JSON into a `prices` table and runs read-only DuckDB SQL. Real work, worth tracking, retrying, and caching. Dispatched from the sandbox. |
| `create_metric`, `create_chart`, `create_table`, `calculate_statistics` | in-process in `analyze` | Microseconds of pure Python; they render HTML blocks into a per-run collector. |
| the model's code | the Monty sandbox | Untrusted LLM code, confined to calling the tools above. |

The split is deliberate. The **fetch is live** — an MCP call the agent makes, not
cached — while the **query is deterministic** given its inputs, so it is the durable,
cached task. And because the sandbox has no `json` or `datetime`, the model never
parses the MCP's raw output; it passes the strings straight to `query`, which does the
reshape where pandas is available.

The agent's `tools` list mixes an `@env.task` (durable dispatch) with plain callables
(in-process), and the MCP server contributes the fetch tool — but the model calls all
of them the same way.

## The report collector

The native code-mode loop ends in a plain-text answer, but the UI renders structured
HTML blocks (metric cards, charts, tables). A per-run `ContextVar` collector bridges
the two: the render tools append their HTML as a side effect and return short
confirmations, and `analyze` reads the blocks back after the agent finishes. The
model then writes a one-line plain-text takeaway that appears beneath the report.

## Files

- `tools.py` - the MCP-output parser and read-only DuckDB `query` (`run_sql`, with a parser-based SELECT-only guard), the report collector, and the render tools (`create_metric`, `create_chart`, `create_table`, `calculate_statistics`).
- `analysis.py` - the durable half: the task environment, the durable `query` task, the Anthropic `call_llm` callback, the Yahoo Finance `MCPServerSpec`, the native `Agent`, and the `analyze` task.
- `app.py` - the `AgentChatAppEnvironment`: theme, prompt nudges, `task_entrypoint=analyze`, `passthrough_auth`, and the deploy entry point.

## The data

There is no local dataset. The agent fetches daily closing prices at runtime from the
`mcp-yahoo-finance` server (public Yahoo Finance data, no credentials). Each
`yf_get_historical_stock_prices` call returns one ticker's closes as a JSON string;
the `query` task parses them into a `prices(ticker, date, close)` table and runs SQL —
moving averages, daily returns, volatility, drawdowns, cross-ticker comparisons.

## Run it

You need a Union deployment and an Anthropic API key stored as a secret.

```bash
flyte create secret anthropic_api_key <your-anthropic-key>
python app.py
```

That deploys the app and its task environment together and prints the app URL. Open
it and ask something like "Compare AAPL and MSFT over the last 6 months." The first
question is slower — the task image builds and the MCP server cold-starts — then each
answer streams progress while the run executes and comes back as a short report plus a
link to the run.

To exercise the analysis half on its own (no app):

```bash
python analysis.py
```
