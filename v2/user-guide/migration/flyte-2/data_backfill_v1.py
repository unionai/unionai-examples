from datetime import date, timedelta

from flytekit import task, workflow, dynamic


@task
def process_day(day: str) -> int:
    # Reprocess a single day's partition; return the row count.
    return len(day)


# @dynamic is needed because the number of days is only known at runtime.
@dynamic
def backfill(start: str, days: int) -> list[int]:
    base = date.fromisoformat(start)
    results = []
    for i in range(days):
        day = (base + timedelta(days=i)).isoformat()
        results.append(process_day(day=day))
    return results


@workflow
def main(start: str, days: int) -> list[int]:
    return backfill(start=start, days=days)
