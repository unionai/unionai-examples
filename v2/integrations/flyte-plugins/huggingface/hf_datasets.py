# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-huggingface",
#    "datasets",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import flyte

image = flyte.Image.from_debian_base(name="huggingface").with_pip_packages("flyteplugins-huggingface")

env = flyte.TaskEnvironment(
    name="huggingface_env",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)
# {{/docs-fragment setup}}

# Point this at a bucket you can read and write to reuse downloads across runs.
# A local path works too, so you can try `flyte run --local` before touching a bucket.
CACHE_ROOT = "s3://my-bucket/flyte-hf-cache"


# {{docs-fragment source}}
import datasets

from flyteplugins.huggingface.datasets import from_hf


@env.task
async def count_reviews(
    ds: datasets.Dataset = from_hf("stanfordnlp/imdb", name="plain_text", split="train"),
) -> int:
    return len(ds)
# {{/docs-fragment source}}


# {{docs-fragment config}}
@env.task
async def count_mrpc(
    ds: datasets.Dataset = from_hf("nyu-mll/glue", name="mrpc", split="train"),
) -> int:
    return len(ds)
# {{/docs-fragment config}}


# {{docs-fragment splits}}
@env.task
async def all_splits_combined(
    ds: datasets.Dataset = from_hf("stanfordnlp/imdb", name="plain_text"),
) -> str:
    # 100,000 rows: train (25k), test (25k) and unsupervised (50k), concatenated.
    # There is no column telling you which split a row came from.
    return f"{len(ds)} rows, columns: {ds.column_names}"
# {{/docs-fragment splits}}


# {{docs-fragment cache}}
@env.task
async def count_reviews_cached(
    ds: datasets.Dataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root=CACHE_ROOT,
    ),
) -> int:
    return len(ds)
# {{/docs-fragment cache}}


# {{docs-fragment projection}}
from collections import OrderedDict
from typing import Annotated


@env.task
async def first_reviews(
    ds: Annotated[datasets.Dataset, OrderedDict(text=str)] = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root=CACHE_ROOT,
    ),
) -> list[str]:
    # `label` was never read off disk.
    return ds["text"][:5]
# {{/docs-fragment projection}}


# {{docs-fragment iterable}}
@env.task
async def add_length(
    ds: datasets.IterableDataset = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root=CACHE_ROOT,
    ),
) -> datasets.IterableDataset:
    def measure(batch):
        batch["length"] = [len(text) for text in batch["text"]]
        return batch

    return ds.map(measure, batched=True)


@env.task
async def mean_length(ds: datasets.IterableDataset, sample: int = 1_000) -> float:
    total = count = 0
    for row in ds.take(sample):
        total += row["length"]
        count += 1
    return total / count
# {{/docs-fragment iterable}}


# {{docs-fragment handoff}}
@env.task
async def build_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {"text": ["hello", "world", "flyte"], "label": [0, 1, 0]}
    )


@env.task
async def keep_positive(ds: datasets.Dataset) -> datasets.Dataset:
    # flatten_indices() is required, not stylistic. filter() only records an
    # index mapping over the original table, and the encoder serializes the
    # underlying table, so without this the task returns every row it was given.
    return ds.filter(lambda row: row["label"] == 1).flatten_indices()
# {{/docs-fragment handoff}}


# {{docs-fragment runtime-arg}}
@env.task
async def count_rows(ds: datasets.Dataset) -> int:
    return len(ds)


@env.task
async def count_any_split(repo: str, split: str) -> int:
    # The dataset is chosen when the parent runs, not when the task is defined.
    return await count_rows(from_hf(repo, split=split, cache_root=CACHE_ROOT))
# {{/docs-fragment runtime-arg}}


# {{docs-fragment passthrough}}
from flyte.io import DataFrame


@env.task
async def route(df: DataFrame) -> DataFrame:
    # Typed as DataFrame, so nothing is downloaded here. The reference is
    # forwarded untouched and resolved by whoever asks for a datasets.Dataset.
    return df
# {{/docs-fragment passthrough}}


# {{docs-fragment main}}
@env.task
async def main() -> str:
    n_train = await count_reviews_cached()
    lengths = await add_length()
    avg = await mean_length(lengths)
    filtered = await keep_positive(await build_dataset())
    return f"{n_train} train reviews, mean length {avg:.1f} chars, {len(filtered)} positive row(s)"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment main}}
