# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b56",
#    "pandas",
#    "pyarrow",
#    "polars",
#    "aiofiles",
# ]
# main = "pandas_driver"
# params = ""
# ///

"""Content-based caching for DataFrame, File, and Dir task inputs.

By default Flyte hashes reference data (DataFrames, files, directories) by its
storage location, not its contents - so a downstream task keyed on such an input
will not see a cache hit even when the data is byte-for-byte identical.

To cache on *content* instead, attach a `HashFunction` to the data at
production time. Flyte then uses that content hash when computing the cache key
of any downstream consuming task.
"""

from typing import Annotated

import pandas as pd
import polars as pl

import flyte
from flyte import Cache
from flyte.io import DataFrame, Dir, File, HashFunction

env = flyte.TaskEnvironment(
    name="content_caching",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow", "polars"),
)

SAMPLE_DATA = {
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "value": [100, 200, 300, 400, 500],
}


# {{docs-fragment pandas}}
def hash_pandas_dataframe(df: pd.DataFrame) -> str:
    # Content-based hash using pandas' built-in row hashing.
    return str(pd.util.hash_pandas_object(df).sum())


# Reusable type alias: a pandas DataFrame whose cache key is its content hash.
HashedPandasDataFrame = Annotated[pd.DataFrame, HashFunction.from_fn(hash_pandas_dataframe)]


@env.task
async def produce_pandas() -> HashedPandasDataFrame:
    # The HashFunction in the return annotation tells Flyte to compute a
    # content hash for this output.
    return pd.DataFrame(SAMPLE_DATA)


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_pandas(df: pd.DataFrame) -> int:
    # Cached on the input's content hash: identical content -> cache hit.
    return int(df["value"].sum())
# {{/docs-fragment pandas}}


# {{docs-fragment polars}}
def hash_polars_dataframe(df: pl.DataFrame) -> str:
    return str(df.hash_rows().sum())


HashedPolarsDataFrame = Annotated[pl.DataFrame, HashFunction.from_fn(hash_polars_dataframe)]


@env.task
async def produce_polars() -> HashedPolarsDataFrame:
    return pl.DataFrame(SAMPLE_DATA)


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_polars(df: pl.DataFrame) -> int:
    return int(df["value"].sum())
# {{/docs-fragment polars}}


# {{docs-fragment flyte-dataframe}}
@env.task
async def produce_flyte_dataframe() -> DataFrame:
    df = pd.DataFrame(SAMPLE_DATA)
    # For flyte.io.DataFrame, pass the HashFunction to `from_local` instead of
    # annotating the return type.
    hash_method = HashFunction.from_fn(hash_pandas_dataframe)
    return await DataFrame.from_local(df, hash_method=hash_method)


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_flyte_dataframe(df: DataFrame) -> int:
    pdf = await df.open(pd.DataFrame).all()
    return int(pdf["value"].sum())
# {{/docs-fragment flyte-dataframe}}


# {{docs-fragment file}}
def hash_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


@env.task
async def produce_file() -> File:
    import aiofiles

    async with aiofiles.open("/tmp/data.csv", "w") as fh:
        await fh.write("id,value\n1,100\n2,200\n")
    # Pass a HashFunction (over the uploaded bytes) to `from_local` - the same
    # mechanism works for `File.new_remote(...)`. The File is then cached on its
    # content rather than its remote path.
    return await File.from_local("/tmp/data.csv", hash_method=HashFunction.from_fn(hash_bytes))


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_file(f: File) -> str:
    async with f.open("rb") as fh:
        return hash_bytes(bytes(await fh.read()))
# {{/docs-fragment file}}


# {{docs-fragment dir}}
@env.task
async def produce_dir() -> Dir:
    import os

    os.makedirs("/tmp/data_dir", exist_ok=True)
    with open("/tmp/data_dir/part.csv", "w") as fh:
        fh.write("id,value\n1,100\n")

    # `Dir.from_local` does not take a HashFunction callable. Instead, compute a
    # content key yourself and pass it as the precomputed `dir_cache_key`.
    content_key = hash_bytes(b"id,value\n1,100\n")
    return await Dir.from_local("/tmp/data_dir/", dir_cache_key=content_key)


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_dir(d: Dir) -> int:
    count = 0
    async for _ in d.walk():
        count += 1
    return count
# {{/docs-fragment dir}}


@env.task
async def pandas_driver() -> bool:
    df = await produce_pandas()
    r1 = await consume_pandas(df)
    r2 = await consume_pandas(df)  # second call hits the cache
    return r1 == r2


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pandas_driver)
    print(run.url)
    run.wait()
