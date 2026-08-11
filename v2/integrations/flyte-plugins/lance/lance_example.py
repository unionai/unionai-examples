# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "flyteplugins-lance",
#    "pandas",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import flyte

image = flyte.Image.from_debian_base(name="lance").with_pip_packages("flyteplugins-lance")

env = flyte.TaskEnvironment(
    name="lance_env",
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)
# {{/docs-fragment setup}}


# {{docs-fragment stream}}
import tempfile

import lance
import pyarrow as pa


@env.task
async def build_dataset(n: int = 10_000) -> lance.LanceDataset:
    uri = f"{tempfile.mkdtemp()}/points.lance"
    lance.write_dataset(pa.table({"id": list(range(n)), "value": [i * i for i in range(n)]}), uri)
    return lance.dataset(uri)


@env.task
async def summarize(ds: lance.LanceDataset) -> dict:
    # `ds` is a live handle. Stream it in batches; nothing is materialized whole.
    total = 0
    for batch in ds.scanner(columns=["value"], batch_size=1024).to_batches():
        total += sum(batch.column("value").to_pylist())
    return {"rows": ds.count_rows(), "sum_of_values": total}
# {{/docs-fragment stream}}


# {{docs-fragment projection}}
@env.task
async def sample_rows(ds: lance.LanceDataset) -> dict:
    # Random access: read only these rows, only these columns.
    rows = ds.take([0, 500, 9_999], columns=["id", "value"]).to_pylist()

    # Predicate pushdown: the filter runs inside Lance, so unmatched rows are
    # never decoded or sent over the wire.
    matched = ds.scanner(columns=["id"], filter="value > 98000000").to_table().num_rows

    return {"sample": rows, "matched": matched}
# {{/docs-fragment projection}}


# {{docs-fragment arrow}}
from collections import OrderedDict
from typing import Annotated

from flyte.io import DataFrame


@env.task
async def build_table() -> Annotated[DataFrame, "lance"]:
    # A bare `DataFrame` or `pa.Table` would be stored as Parquet. The "lance"
    # annotation is what selects this plugin's encoder.
    table = pa.table(
        {
            "city": ["NYC", "SF", "LA", "SEA"],
            "temp_c": [7, 15, 20, 11],
            "humidity": [55, 70, 40, 80],
        }
    )
    return DataFrame.wrap_df(table)


@env.task
async def read_as_dataset(ds: lance.LanceDataset) -> int:
    # The same stored bytes, opened lazily as a streaming handle.
    return ds.count_rows()


@env.task
async def read_as_table(table: Annotated[pa.Table, OrderedDict(city=str, temp_c=int)]) -> dict:
    # Decoded eagerly into memory, narrowed to the two annotated columns.
    return {"columns": table.column_names, "rows": table.num_rows}
# {{/docs-fragment arrow}}


# {{docs-fragment interop}}
# `to_pandas()` is a pyarrow method, but it still needs pandas installed in the
# task image: .with_pip_packages("flyteplugins-lance", "pandas")


@env.task
async def as_pandas(table: pa.Table) -> int:
    """Convert inside the task. Declaring `pd.DataFrame` directly would fail:
    the plugin registers no lance-to-pandas handler."""
    df = table.to_pandas()
    return len(df)


@env.task
async def as_pandas_streaming(ds: lance.LanceDataset) -> int:
    """Convert per batch instead, so the dataset is never held whole."""
    rows = 0
    for batch in ds.scanner(columns=["id"], batch_size=1024).to_batches():
        rows += len(batch.to_pandas())
    return rows
# {{/docs-fragment interop}}


# {{docs-fragment reference}}
import os


@env.task
async def convert(n: int = 10_000, chunk: int = 2_000) -> DataFrame:
    """Write a Lance dataset in bounded chunks, then hand it off by reference."""
    uri = os.path.join(tempfile.mkdtemp(), "dataset.lance")
    mode = "create"
    for start in range(0, n, chunk):
        rows = list(range(start, min(start + chunk, n)))
        lance.write_dataset(pa.table({"id": rows}), uri, mode=mode)
        mode = "append"

    # Flyte uploads the .lance directory verbatim: no re-read, no re-encode.
    return DataFrame(uri=uri, format="lance")


@env.task
async def inspect(ds: lance.LanceDataset) -> dict:
    # The consumer still takes the raw Lance type; the "lance" format resolves
    # the handoff regardless of which form the producer returned.
    return {"rows": ds.count_rows(), "fragments": len(ds.get_fragments())}
# {{/docs-fragment reference}}


# {{docs-fragment fragments}}
from functools import partial


@env.task
async def scan_fragment(df: DataFrame, fragment_id: int) -> int:
    """Read exactly one fragment. Each worker touches a disjoint slice of files."""
    ds = await df.open(lance.LanceDataset).all()
    return ds.get_fragment(fragment_id).to_table(columns=["id"]).num_rows


@env.task
async def fan_out(df: DataFrame) -> int:
    # Pass the dataset as a `DataFrame`, not a `lance.LanceDataset`: the reference
    # form keeps the stored bytes (and therefore the fragment ids) identical for
    # every worker. Returning a `lance.LanceDataset` would re-encode the dataset
    # at the boundary and coalesce the fragments, invalidating these ids.
    ds = await df.open(lance.LanceDataset).all()
    fragment_ids = [f.fragment_id for f in ds.get_fragments()]

    total = 0
    async for count in flyte.map.aio(partial(scan_fragment, df), fragment_ids):
        if isinstance(count, Exception):
            raise count
        total += count
    return total
# {{/docs-fragment fragments}}


# {{docs-fragment main}}
@env.task
async def main(n: int = 10_000) -> dict:
    ds = await build_dataset(n)

    table_df = await build_table()

    referenced = await convert(n)

    return {
        "summary": await summarize(ds),
        "sampled": await sample_rows(ds),
        "arrow_streaming_rows": await read_as_dataset(table_df),
        "arrow_eager": await read_as_table(table_df),
        "pandas_rows": await as_pandas(table_df),
        "pandas_streaming_rows": await as_pandas_streaming(referenced),
        "referenced": await inspect(referenced),
        "fanned_out_rows": await fan_out(referenced),
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment main}}
