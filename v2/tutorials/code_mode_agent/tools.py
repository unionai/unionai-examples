"""Tools and dataset for the Code Mode analytics agent.

The agent is given a small set of tools. It writes Python *orchestration* code
that calls them; that code runs in the Monty sandbox, which allows no imports,
no IO, and no network, so the only things the generated code can touch are the
tools registered here.

Two kinds of tools, on purpose:

* ``query`` runs DuckDB SQL over the dataset. In ``analysis.py`` it is wrapped as
  a durable ``@env.task``, so every query the model writes becomes a tracked,
  retryable Flyte task. This is the heavy tool.
* ``create_metric``, ``create_chart``, ``create_table``, and
  ``calculate_statistics`` are cheap, pure-Python helpers. They run in-process,
  with no durability overhead. The ``create_*`` ones return HTML the UI renders;
  together they let the model assemble a small report rather than a lone chart.

To add a tool: write a function with type annotations and a docstring, then add
it to the registry in ``analysis.py``. The agent regenerates its system prompt
from the signatures and docstrings, so there is nothing else to wire up.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json as _json
import math
import random
from functools import lru_cache

# ---------------------------------------------------------------------------
# Chart palette (cool blues, on a light UI)
# ---------------------------------------------------------------------------

CHART_COLORS = [
    "rgba(14, 165, 233, 0.8)",  # #0ea5e9 — sky
    "rgba(37, 99, 235, 0.8)",  # #2563eb — blue
    "rgba(6, 182, 212, 0.8)",  # #06b6d4 — cyan
    "rgba(99, 102, 241, 0.8)",  # #6366f1 — indigo
    "rgba(8, 145, 178, 0.8)",  # #0891b2 — deep cyan
]

CHART_BORDERS = ["#0ea5e9", "#2563eb", "#06b6d4", "#6366f1", "#0891b2"]

# ---------------------------------------------------------------------------
# Dataset — a stand-in "orders" table queried with DuckDB
# ---------------------------------------------------------------------------
# This is generated deterministically so the example runs with no external data
# to fetch. It stands in for a real table: to query your own data, change
# `_dataframe()` to read a file or a warehouse, e.g.
#   duckdb.read_parquet("s3://bucket/orders.parquet")
# The rest of the agent is unchanged — the model still writes SQL against a table
# called `orders`.

# This description is injected into the system prompt so the model knows the
# schema without a round-trip.
DATASET_DESCRIPTION = (
    "You are querying a DuckDB table named `orders` (ecommerce orders for 2024).\n"
    "Columns:\n"
    "  - order_date  DATE     (2024-01-01 .. 2024-12-31)\n"
    "  - region      TEXT     (North, South, East, West)\n"
    "  - category    TEXT     (Electronics, Furniture, Stationery, Outdoor)\n"
    "  - channel     TEXT     (Web, Mobile, Store)\n"
    "  - revenue     DOUBLE   (order revenue in USD)\n"
    "  - units       INTEGER  (items in the order)\n"
    "  - is_returned BOOLEAN  (whether the order was returned)\n"
    "About 6,000 rows. Prefer aggregating in SQL (GROUP BY, SUM, AVG) and let the\n"
    "chart tool render the result."
)

_REGIONS = ["North", "South", "East", "West"]
_CATEGORIES = {"Electronics": 220.0, "Furniture": 180.0, "Stationery": 12.0, "Outdoor": 95.0}
_CHANNELS = ["Web", "Mobile", "Store"]
_MONTH_SEASONALITY = [0.85, 0.88, 0.95, 1.0, 1.05, 1.10, 1.08, 1.12, 1.06, 1.02, 1.15, 1.25]


@lru_cache(maxsize=1)
def _rows() -> list:
    """Build the demo orders as a list of row dicts once (deterministic, pure Python).

    Kept free of pandas/duckdb so a lightweight preview (see `dataset_sample`) can read a
    few rows without importing the query engine.
    """
    rng = random.Random(7)
    start = _dt.date(2024, 1, 1)
    rows: list[dict] = []
    for _ in range(6000):
        day = start + _dt.timedelta(days=rng.randint(0, 364))
        category = rng.choice(list(_CATEGORIES))
        region = rng.choice(_REGIONS)
        channel = rng.choice(_CHANNELS)
        units = rng.randint(1, 8)
        unit_price = _CATEGORIES[category] * rng.uniform(0.8, 1.3)
        revenue = round(units * unit_price * _MONTH_SEASONALITY[day.month - 1], 2)
        rows.append(
            {
                "order_date": day,
                "region": region,
                "category": category,
                "channel": channel,
                "revenue": revenue,
                "units": units,
                "is_returned": rng.random() < 0.06,
            }
        )
    return rows


@lru_cache(maxsize=1)
def _dataframe():
    """The orders as a pandas DataFrame (built once), for the DuckDB query path."""
    import pandas as pd

    return pd.DataFrame(_rows())


# {{docs-fragment dataset_sample}}
@lru_cache(maxsize=1)
def dataset_sample(n: int = 12) -> tuple:
    """A cheap sample for the UI's data preview: ``(headers, rows, total_count)``.

    Reads the row dicts directly (no pandas, no duckdb, no run), so the app can show what
    the data looks like before anyone queries it. Cached, so it is built once.
    """
    rows = _rows()
    headers = list(rows[0].keys())
    sample = [[_jsonable(rows[i][h]) for h in headers] for i in range(min(n, len(rows)))]
    return headers, sample, len(rows)
# {{/docs-fragment dataset_sample}}


def _jsonable(value: object) -> object:
    """Coerce DuckDB scalars to JSON-friendly Python types."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


# {{docs-fragment query_tool}}
async def query(sql: str) -> list:
    """Run a read-only SQL query over the `orders` table and return rows.

    Args:
        sql: A DuckDB SELECT statement against the table `orders`
             (columns: order_date, region, category, channel, revenue, units,
             is_returned). Aggregate in SQL where you can.

    Returns:
        A list of row dicts (one per result row), with dates as ISO strings.
    """
    import duckdb

    con = duckdb.connect()
    con.register("orders", _dataframe())
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
    median = ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
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
    """Generate a self-contained Chart.js HTML snippet.

    Args:
        chart_type: One of "bar", "line", "pie", "doughnut".
        title: Chart title displayed above the canvas.
        labels: X-axis labels (or slice labels for pie/doughnut).
        values: Either a flat list of numbers, or a list of
                {"label": str, "data": list[number]} dicts for multi-series.

    Returns:
        HTML string with a <canvas> whose Chart.js config rides on a data attribute;
        the UI instantiates it (no element id, so repeated/cached HTML never collides).
    """
    if not values:
        return (
            '<div class="block" style="margin:16px 0;color:#64748b;font-size:13px;">'
            f"(no data to chart for “{title}”)</div>"
        )

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
                "backgroundColor": bg if chart_type in ("pie", "doughnut") else CHART_COLORS[0],
                "borderColor": border if chart_type in ("pie", "doughnut") else CHART_BORDERS[0],
                "borderWidth": 2,
                "tension": 0.3,
                "fill": chart_type == "line",
            }
        ]

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": title, "font": {"size": 16}}},
        },
    }

    # Carry the chart config on the element itself (no id, no inline <script>). The UI
    # instantiates each canvas by element reference, so identical HTML — e.g. a repeated,
    # *cached* answer rendered in a new bubble — can never collide on a shared id.
    config_json = _html.escape(_json.dumps(config), quote=True)
    return (
        '<div class="block chart-block" style="position:relative;height:340px;margin:18px 0;">'
        f'<canvas class="cm-chart" data-config="{config_json}"></canvas></div>'
    )


async def create_metric(label: str, value: str, delta: str = "") -> str:
    """Render a single KPI card (a big number with a label).

    Use for headline figures, e.g. total revenue or average order value. Group several
    by returning them next to each other; they lay out in a row.

    Args:
        label: Short caption, e.g. "Total revenue".
        value: The formatted value to display, e.g. "$1.2M" or "27%".
        delta: Optional change note, e.g. "+8% vs last month".

    Returns:
        HTML string for one metric card.
    """
    delta_html = (
        f'<div style="font-size:12px;color:#64748b;margin-top:4px;">{delta}</div>' if delta else ""
    )
    return (
        '<div class="block metric-card" style="display:inline-block;min-width:150px;margin:8px 10px 8px 0;'
        'padding:16px 20px;background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.25);'
        'border-radius:14px;vertical-align:top;">'
        f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>'
        f'<div style="font-size:26px;font-weight:700;color:#0284c7;margin-top:4px;">{value}</div>'
        f"{delta_html}</div>"
    )


async def create_table(title: str, headers: list, rows: list) -> str:
    """Render a data table.

    Use for tabular breakdowns (e.g. top products, per-region detail) where a chart
    would lose the exact numbers.

    Args:
        title: Table caption shown above it.
        headers: Column names.
        rows: List of rows, each a list of cell values (same length as headers).

    Returns:
        HTML string for the table.
    """
    head = "".join(
        f'<th style="text-align:left;padding:8px 12px;color:#0284c7;border-bottom:1px solid '
        f'rgba(14,165,233,0.3);">{h}</th>'
        for h in headers
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:8px 12px;border-bottom:1px solid rgba(15,23,42,0.08);">{c}</td>'
            for c in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="block table-block" style="margin:16px 0;overflow-x:auto;">'
        f'<div style="font-size:13px;color:#64748b;margin-bottom:8px;">{title}</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )
