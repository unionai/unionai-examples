# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "flyteplugins-lance",
#    "pylance",
#    "pyarrow",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import random
import time
from typing import Annotated

import flyte
import flyte.report
import lance
import pyarrow as pa
from flyte.io import DataFrame

image = flyte.Image.from_debian_base(name="lance-benchmark").with_pip_packages("flyteplugins-lance")

env = flyte.TaskEnvironment(
    name="lance_benchmark",
    image=image,
    resources=flyte.Resources(cpu="2", memory="8Gi"),
)


def build_table(n_rows: int, payload_bytes: int) -> pa.Table:
    """An id, a float feature, and a binary payload standing in for an embedding."""
    rng = random.Random(0)
    return pa.table(
        {
            "id": pa.array(range(n_rows), type=pa.int64()),
            "x": pa.array([rng.random() for _ in range(n_rows)], type=pa.float64()),
            "payload": pa.array([rng.randbytes(payload_bytes) for _ in range(n_rows)], type=pa.large_binary()),
        }
    )
# {{/docs-fragment setup}}


# {{docs-fragment producers}}
@env.task
async def write_parquet(n_rows: int = 100_000, payload_bytes: int = 512) -> DataFrame:
    """No format annotation, so this is stored as Parquet (Flyte's default)."""
    return DataFrame.wrap_df(build_table(n_rows, payload_bytes))


@env.task
async def write_lance(n_rows: int = 100_000, payload_bytes: int = 512) -> Annotated[DataFrame, "lance"]:
    """The same rows, stored as Lance by this plugin."""
    return DataFrame.wrap_df(build_table(n_rows, payload_bytes))
# {{/docs-fragment producers}}


def _best_of(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


async def _best_of_async(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        await fn()
        best = min(best, time.perf_counter() - start)
    return best


# {{docs-fragment compare}}
@env.task(report=True)
async def compare(parquet_df: DataFrame, lance_ds: lance.LanceDataset, n_random_rows: int = 1_000) -> dict:
    n = lance_ds.count_rows()
    indices = random.Random(7).sample(range(n), k=min(n_random_rows, n))
    columns = ["id", "x", "payload"]

    # Lance: fetch exactly the requested rows off the open handle.
    lance_bytes = lance_ds.take(indices, columns=columns).nbytes
    lance_seconds = _best_of(lambda: lance_ds.take(indices, columns=columns))

    # Parquet: the decoder is eager, so getting any subset means materializing
    # the whole table first, then indexing into it in memory.
    async def parquet_random():
        table = await parquet_df.open(pa.Table).all()
        table.take(indices)

    parquet_bytes = (await parquet_df.open(pa.Table).all()).nbytes
    parquet_seconds = await _best_of_async(parquet_random)

    await flyte.report.replace.aio(
        _render(n, len(indices), parquet_bytes, lance_bytes, parquet_seconds, lance_seconds),
        do_flush=True,
    )

    return {
        "rows_in_dataset": n,
        "rows_requested": len(indices),
        "parquet_mb_read": round(parquet_bytes / 1e6, 1),
        "lance_mb_read": round(lance_bytes / 1e6, 2),
        "less_data_read_x": round(parquet_bytes / lance_bytes, 1),
        "parquet_ms": round(parquet_seconds * 1e3, 1),
        "lance_ms": round(lance_seconds * 1e3, 1),
        "faster_x": round(parquet_seconds / lance_seconds, 1),
    }
# {{/docs-fragment compare}}


def _bar(label: str, value: float, largest: float, unit: str, color: str) -> str:
    pct = 100 * value / largest if largest else 0
    return (
        '<div style="margin:10px 0">'
        f'<div style="font-size:13px;margin-bottom:4px">{label} &mdash; <b>{value:,.1f}{unit}</b></div>'
        '<div style="background:#e9ecef;border-radius:5px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;background:{color};height:18px"></div></div></div>'
    )


def _render(n, k, parquet_bytes, lance_bytes, parquet_seconds, lance_seconds) -> str:
    parquet_mb, lance_mb = parquet_bytes / 1e6, lance_bytes / 1e6
    largest = max(parquet_mb, lance_mb)
    return f"""
    <div style="font-family:system-ui,sans-serif;max-width:720px">
      <h2>Fetching {k:,} scattered rows from {n:,}</h2>
      <p style="color:#444">Same columnar data, same <code>flyte.io.DataFrame</code> interface.
      What differs is how much has to be read to answer the request.</p>
      <div style="margin:14px 0;padding:12px 14px;background:#e6fcf5;border-radius:8px;color:#087f5b">
        Parquet materialized <b>{parquet_mb:,.0f} MB</b> to return {k:,} rows.
        Lance read <b>{lance_mb:,.1f} MB</b> &mdash;
        <b>{parquet_bytes / lance_bytes:,.0f}x less data</b>.
      </div>
      <h3 style="margin:20px 0 2px">Data read (lower is better)</h3>
      {_bar("Parquet", parquet_mb, largest, " MB", "#adb5bd")}
      {_bar("Lance", lance_mb, largest, " MB", "#06d6a0")}
      <p style="color:#666;font-size:13px;margin-top:16px">
        Fetch time: Parquet {parquet_seconds * 1e3:,.0f} ms vs Lance {lance_seconds * 1e3:,.1f} ms.
        The bytes-read ratio is the durable number &mdash; it is cache-independent, and it is what
        turns into a latency gap once the data sits on object storage rather than local disk.
      </p>
    </div>
    """


# {{docs-fragment main}}
@env.task
async def main(n_rows: int = 100_000, payload_bytes: int = 512, n_random_rows: int = 1_000) -> dict:
    parquet_df = await write_parquet(n_rows, payload_bytes)
    lance_df = await write_lance(n_rows, payload_bytes)
    return await compare(parquet_df, lance_df, n_random_rows)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment main}}
