"""Tools and data access for the Code Mode stock-analysis agent.

The agent (``flyte.ai.agents.Agent`` in ``code_mode``) writes Python
orchestration code that calls these tools; that code runs in the Monty
sandbox, which allows no imports, no IO, and no network, so the only things the
generated code can touch are the tools registered in ``analysis.py``.

Two kinds of tools, on purpose:

* The **fetch** is a Yahoo Finance MCP tool (``yf_get_historical_stock_prices``),
  registered on the agent via ``mcp_servers`` in ``analysis.py``. It is the only
  path to the network — the sandbox has none — so it is the agent's live data
  source. It returns a raw JSON *string* of closing prices; the sandbox does not
  parse it (it has no ``json``), it just hands it to ``query``.
* ``query`` runs read-only DuckDB SQL over the fetched series. In ``analysis.py``
  it is a durable ``@env.task``, so the heavy analytics (moving averages,
  volatility, drawdowns, cross-ticker joins) run as a tracked, cached Flyte task.
  It parses the raw MCP strings into a ``prices`` table before running the SQL —
  the messy reshape lives here, where pandas is available, not in the sandbox.
* ``create_metric``, ``create_chart``, ``create_table``, and
  ``calculate_statistics`` are cheap, pure-Python helpers that run in-process.
  The ``create_*`` ones render HTML blocks into a per-run report collector.

To add a tool: write a function with type annotations and a docstring, then add
it to the agent's ``tools`` list in ``analysis.py``. The agent generates its
system prompt from the signatures and docstrings, so there is nothing else to
wire up.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json as _json
import math
import uuid

CHART_COLORS = [
    "rgba(14, 165, 233, 0.8)",  # #0ea5e9 — sky
    "rgba(37, 99, 235, 0.8)",  # #2563eb — blue
    "rgba(6, 182, 212, 0.8)",  # #06b6d4 — cyan
    "rgba(99, 102, 241, 0.8)",  # #6366f1 — indigo
    "rgba(8, 145, 178, 0.8)",  # #0891b2 — deep cyan
]

CHART_BORDERS = ["#0ea5e9", "#2563eb", "#06b6d4", "#6366f1", "#0891b2"]

# Dataset — live stock closing prices, fetched via the Yahoo Finance MCP server
#
# There is no local data to fetch: the agent pulls prices at runtime from the
# `mcp-yahoo-finance` server (registered in `analysis.py`). This description is
# injected into the system prompt so the model knows how the two heavy tools fit
# together without a round-trip.

DATA_DESCRIPTION = (
    "You analyze daily stock closing prices. There are two heavy tools.\n"
    "\n"
    "Fetching (one ticker per call, via the Yahoo Finance MCP server):\n"
    "  yf_get_historical_stock_prices(symbol=..., period='1y', interval='1d')\n"
    "  returns a JSON *string* of closing prices keyed by timestamp. Do NOT parse\n"
    "  it in your code — the sandbox has no json or datetime module. Pass the\n"
    "  string straight to query(). Call it once per ticker (await each call) and\n"
    "  collect the returned strings into a dict for query(). Valid period: 1mo,\n"
    "  3mo, 6mo, 1y, 2y, 5y, ytd, max. Valid interval: 1d, 1wk, 1mo.\n"
    "\n"
    "Analyzing (durable DuckDB task):\n"
    "  query(sql, series) where `series` maps each ticker symbol to the JSON\n"
    "  string returned by yf_get_historical_stock_prices for it. The task parses\n"
    "  those into one table:\n"
    "     prices(ticker TEXT, date DATE, close DOUBLE)\n"
    "  Write a single read-only SELECT against `prices`. Do the math in SQL:\n"
    "  window functions (AVG(...) OVER (PARTITION BY ticker ORDER BY date ...))\n"
    "  for moving averages, LAG(...) for daily returns, STDDEV for volatility,\n"
    "  and GROUP BY / self-joins for cross-ticker comparisons."
)


def _jsonable(value: object) -> object:
    """Coerce DuckDB scalars to JSON-friendly Python types."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    return value


# {{docs-fragment collector}}
# The native code-mode loop ends in a plain-text answer, but the UI renders
# structured HTML blocks. A per-run collector bridges the two: each render tool
# appends its HTML here as a side effect, and the `analyze` task reads the blocks
# back after the agent finishes. A ContextVar keeps concurrent runs isolated.
_REPORT: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "report", default=None
)


def start_report() -> None:
    """Begin a fresh report for this run (called by `analyze` before the agent)."""
    _REPORT.set([])


def collect_report() -> list[str]:
    """Return the HTML blocks rendered so far, in the order they were created."""
    return list(_REPORT.get() or [])


def _add_block(html: str) -> None:
    blocks = _REPORT.get()
    if blocks is not None:
        blocks.append(html)


# {{/docs-fragment collector}}


# {{docs-fragment sql_guard}}
# The tool is a safety boundary. The model can only call the tools you register, so
# narrowing what a tool accepts shrinks the blast radius. `query` allows a single
# read-only SELECT and nothing else. DuckDB's own parser classifies the statement, so
# there is no brittle keyword matching to trip over identifiers or string literals.
def _ensure_read_only(con, sql: str) -> None:
    import duckdb

    statements = con.extract_statements(sql)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError("Only a single read-only SELECT query is allowed.")


# {{/docs-fragment sql_guard}}


# {{docs-fragment query_tool}}
async def run_sql(sql: str, series: dict[str, str]) -> list:
    """Parse raw Yahoo Finance price JSON per ticker, then run a read-only query.

    Args:
        sql: A DuckDB SELECT statement against the table `prices`
             (columns: ticker, date, close). Aggregate in SQL where you can.
        series: Maps ticker symbol -> the JSON string returned by
                yf_get_historical_stock_prices for it (closing prices keyed by
                epoch-millisecond timestamp).

    Returns:
        A list of row dicts (one per result row), with dates as ISO strings.
    """
    import duckdb
    import pandas as pd

    # Parse each ticker's raw MCP payload into rows and stack them into one table.
    # This reshape needs json + pandas, which the Monty sandbox lacks — so it runs
    # here, in the durable task, not in the model's generated code.
    frames = []
    for ticker, raw in series.items():
        data = _json.loads(raw) if raw else {}
        if not data:
            continue
        frame = pd.DataFrame({"ts": list(data.keys()), "close": list(data.values())})
        # The MCP keys its close prices by timestamp, but the format varies by
        # pandas version inside the server: ISO date strings ("2025-07-03") or
        # epoch-millisecond integers. Detect which and parse accordingly.
        ts = frame["ts"].astype(str)
        if ts.str.fullmatch(r"\d+").all():
            frame["date"] = pd.to_datetime(ts.astype("int64"), unit="ms").dt.date
        else:
            frame["date"] = pd.to_datetime(ts).dt.date
        frame["ticker"] = ticker
        frames.append(frame[["ticker", "date", "close"]])

    prices = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["ticker", "date", "close"])
    )

    # Lock the engine down: no reading or writing files, no extensions, no network.
    con = duckdb.connect(config={"enable_external_access": "false"})
    _ensure_read_only(con, sql)

    con.register("prices", prices)
    rel = con.execute(sql)
    columns = [d[0] for d in rel.description]
    return [{c: _jsonable(v) for c, v in zip(columns, row)} for row in rel.fetchall()]


# {{/docs-fragment query_tool}}


async def calculate_statistics(rows: list, column: str) -> dict:
    """Calculate descriptive statistics for a numeric column of query rows.

    Args:
        rows: A list of row dicts, e.g. the output of query().
        column: Name of the numeric column to analyze.

    Returns:
        Dict with keys: count, mean, median, min, max, std_dev.
    """
    vals = [row[column] for row in rows if column in row and row[column] is not None]
    if not vals:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    n = len(vals)
    mean = sum(vals) / n
    ordered = sorted(vals)
    median = (
        ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    )
    variance = sum((v - mean) ** 2 for v in vals) / n
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": min(vals),
        "max": max(vals),
        "std_dev": round(math.sqrt(variance), 2),
    }


async def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    """Add a chart to the report (rendered with Chart.js in the UI).

    Blocks appear in the report in the order the create_* tools are called.

    Args:
        chart_type: One of "bar", "line", "pie", "doughnut".
        title: Chart title displayed above the canvas.
        labels: X-axis labels (or slice labels for pie/doughnut).
        values: Either a flat list of numbers, or a list of
                {"label": str, "data": list[number]} dicts for multi-series.

    Returns:
        A short confirmation string.
    """
    if not values:
        return f"chart {title!r} skipped: no data to plot"

    if isinstance(values[0], dict):
        datasets = []
        for i, series in enumerate(values):
            idx = i % len(CHART_COLORS)
            datasets.append(
                {
                    "label": series["label"],
                    "data": series["data"],
                    "backgroundColor": CHART_COLORS[idx],
                    "borderColor": CHART_BORDERS[idx],
                    "borderWidth": 2,
                    "tension": 0.3,
                    "fill": False,
                }
            )
    else:
        bg = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(values))]
        border = [CHART_BORDERS[i % len(CHART_BORDERS)] for i in range(len(values))]
        datasets = [
            {
                "label": title,
                "data": values,
                "backgroundColor": (
                    bg if chart_type in ("pie", "doughnut") else CHART_COLORS[0]
                ),
                "borderColor": (
                    border if chart_type in ("pie", "doughnut") else CHART_BORDERS[0]
                ),
                "borderWidth": 2,
                "tension": 0.3,
                "fill": chart_type == "line",
            }
        ]

    # Light text and faint grid lines so the chart reads on the chat UI's dark theme
    # (Chart.js defaults to dark grey text, which disappears on a near-black page).
    options: dict = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "title": {
                "display": True,
                "text": title,
                "font": {"size": 16},
                "color": "#e5e7eb",
            },
            "legend": {"labels": {"color": "#cbd5e1"}},
        },
    }
    if chart_type in ("bar", "line"):
        options["scales"] = {
            axis: {
                "ticks": {"color": "#94a3b8"},
                "grid": {"color": "rgba(148,163,184,0.15)"},
            }
            for axis in ("x", "y")
        }
    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": options,
    }

    # A self-contained canvas plus the script that instantiates it. The chat UI injects
    # each block's HTML and re-runs its <script>, so the chart draws itself. A unique id
    # keeps two charts in one report (or a repeated title) from colliding, and escaping
    # </ stops any string in the config from closing the <script> early.
    canvas_id = "cm-chart-" + uuid.uuid4().hex[:8]
    config_json = _json.dumps(config).replace("</", "<\\/")
    _add_block(
        '<div class="block chart-block" style="position:relative;height:340px;margin:18px 0;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f"<script>try{{new Chart(document.getElementById('{canvas_id}'),"
        f"{config_json});}}catch(e){{console.error('chart {canvas_id}',e);}}</script>"
    )
    return f"chart {title!r} added to the report"


async def create_metric(label: str, value: str, delta: str = "") -> str:
    """Add a single KPI card (a big number with a label) to the report.

    Use for headline figures, e.g. latest price or period return. Consecutive
    metric cards lay out in a row. Blocks appear in the order the tools are called.

    Args:
        label: Short caption, e.g. "AAPL return".
        value: The formatted value to display, e.g. "$185.64" or "+12%".
        delta: Optional change note, e.g. "+8% vs last month".

    Returns:
        A short confirmation string.
    """
    # Always render the delta line (blank when there is no delta) so every card is the
    # same height whether or not a delta was passed, and a row of cards stays aligned.
    # Colors are tuned for the chat UI's dark theme.
    delta_html = f'<div style="font-size:12px;color:#94a3b8;margin-top:4px;">{delta or "&nbsp;"}</div>'
    _add_block(
        '<div class="block metric-card" style="display:inline-block;min-width:150px;margin:8px 10px 8px 0;'
        "padding:16px 20px;background:rgba(14,165,233,0.12);border:1px solid rgba(14,165,233,0.35);"
        'border-radius:14px;vertical-align:top;">'
        f'<div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>'
        f'<div style="font-size:26px;font-weight:700;color:#7dd3fc;margin-top:4px;">{value}</div>'
        f"{delta_html}</div>"
    )
    return f"metric {label!r} added to the report"


async def create_table(title: str, headers: list, rows: list) -> str:
    """Add a data table to the report.

    Use for tabular breakdowns (e.g. per-ticker detail) where a chart would lose
    the exact numbers. Blocks appear in the order the tools are called.

    Args:
        title: Table caption shown above it.
        headers: Column names.
        rows: List of rows, each a list of cell values (same length as headers).

    Returns:
        A short confirmation string.
    """
    # Colors tuned for the chat UI's dark theme.
    head = "".join(
        f'<th style="text-align:left;padding:8px 12px;color:#7dd3fc;border-bottom:1px solid '
        f'rgba(14,165,233,0.4);">{h}</th>'
        for h in headers
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(148,163,184,0.15);">{c}</td>'
            for c in row
        )
        + "</tr>"
        for row in rows
    )
    _add_block(
        '<div class="block table-block" style="margin:16px 0;overflow-x:auto;">'
        f'<div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">{title}</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )
    return f"table {title!r} added to the report ({len(rows)} rows)"
