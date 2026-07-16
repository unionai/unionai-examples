# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "start=2024-01-01 days=3"
# ///

# {{docs-fragment all}}
import asyncio
from datetime import date, timedelta

import flyte

env = flyte.TaskEnvironment(name="data_backfill")


@env.task
async def process_day(day: str) -> int:
    # Reprocess a single day's partition; return the row count.
    return len(day)


# A plain task builds the date range at runtime and fans the days out in
# parallel with asyncio.gather -- no @dynamic and no map_task needed.
@env.task
async def main(start: str, days: int) -> list[int]:
    base = date.fromisoformat(start)
    coros = [
        process_day((base + timedelta(days=i)).isoformat())
        for i in range(days)
    ]
    return list(await asyncio.gather(*coros))
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, start="2024-01-01", days=3)
    print(r.name)
    print(r.url)
    r.wait()
