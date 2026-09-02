"""A toy data-processing workflow — developed (and debugged) by an agent.

This is the workflow the agent in `agent_task.py` runs, observes, patches, and forks until it
passes. It processes mock sales records through three steps:

    load_records -> clean_records -> summarize

It ships with two planted bugs, one per downstream step, so the agent's
observe -> patch -> fork loop has real work to do — and so each fork reuses everything
upstream of the fix:

  * `clean_records` reads a `price` field that does not exist — the field is called
    `unit_price` — so it dies with `KeyError: 'price'` on the first record.
  * `summarize` treats the per-region averages as dicts when they are floats, and dies with
    `TypeError: 'float' object is not subscriptable`.

This file is intentionally a plain module: it is launched by the agent (and forked with the
agent's patches), not run directly.
"""

import asyncio
import random

import flyte

env = flyte.TaskEnvironment(
    name="toy_sales_pipeline",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)

REGIONS = ("AMER", "EMEA", "APAC")


# {{docs-fragment workflow}}
@env.task
async def load_records(n_records: int, seed: int) -> list[dict]:
    """Ingest mock sales records (stand-in for a slow data source)."""
    rng = random.Random(seed)
    records = [
        {
            "id": i,
            "region": rng.choice(REGIONS),
            "units": rng.randint(1, 100),
            "unit_price": round(rng.uniform(5.0, 50.0), 2),
        }
        for i in range(n_records)
    ]
    print(f"load_records: ingested {n_records} records")
    await asyncio.sleep(2)  # stand-in for a real ingestion cost
    return records


@env.task
async def clean_records(records: list[dict]) -> list[dict]:
    """Derive per-record revenue."""
    cleaned = []
    for record in records:
        record = dict(record)
        # BUG 1: the field is called "unit_price", not "price" -> KeyError on every record.
        record["revenue"] = record["units"] * record["price"]
        cleaned.append(record)
    print(f"clean_records: derived revenue for {len(cleaned)} records")
    await asyncio.sleep(2)
    return cleaned


@env.task
async def summarize(records: list[dict]) -> dict:
    """Average revenue per region and pick the top region."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        region = record["region"]
        totals[region] = totals.get(region, 0.0) + record["revenue"]
        counts[region] = counts.get(region, 0) + 1
    averages = {region: totals[region] / counts[region] for region in totals}
    # BUG 2: the averages are floats, not dicts -> TypeError: 'float' is not subscriptable.
    top_region, top_average = max(averages.items(), key=lambda kv: kv[1]["total"])
    summary = {
        "average_revenue_by_region": averages,
        "top_region": top_region,
        "top_region_average_revenue": top_average,
    }
    print(f"summarize: {summary}")
    await asyncio.sleep(2)
    return summary


@env.task
async def main(n_records: int = 50, seed: int = 7) -> dict:
    """The whole toy pipeline."""
    records = await load_records(n_records, seed)
    cleaned = await clean_records(records)
    return await summarize(cleaned)
# {{/docs-fragment workflow}}
