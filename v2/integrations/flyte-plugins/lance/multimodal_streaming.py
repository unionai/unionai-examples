# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "flyteplugins-lance",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import os
import random
import tempfile

import flyte
import lance
import pyarrow as pa
from flyte.io import DataFrame

image = flyte.Image.from_debian_base(name="lance-multimodal").with_pip_packages("flyteplugins-lance")

env = flyte.TaskEnvironment(
    name="lance_multimodal",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)

# `lance-encoding:blob` keeps large values out of the regular column layout, so a
# scan that doesn't ask for `image` never pays for the image bytes.
SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("image", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
        pa.field("label", pa.int32()),
    ]
)
# {{/docs-fragment setup}}


def _fake_image_bytes(i: int) -> bytes:
    """Stand-in for encoded PNG/JPEG bytes, deliberately non-uniform in size."""
    return f"IMG{i}".encode() * (i % 7 + 1) * 64


# {{docs-fragment convert}}
@env.task
async def convert(n: int = 4_000, chunk: int = 512) -> DataFrame:
    """Fold many small samples into one Lance dataset, a chunk at a time."""
    uri = os.path.join(tempfile.mkdtemp(), "images.lance")

    mode = "create"
    for start in range(0, n, chunk):
        rows = list(range(start, min(start + chunk, n)))
        table = pa.table(
            {
                "id": rows,
                "image": [_fake_image_bytes(i) for i in rows],
                "label": [i % 10 for i in rows],
            },
            schema=SCHEMA,
        )
        lance.write_dataset(table, uri, mode=mode)
        mode = "append"

    # Hand off by reference. Returning a `lance.LanceDataset` here would re-encode
    # the dataset through Arrow, which a blob-encoded column does not survive.
    return DataFrame(uri=uri, format="lance")
# {{/docs-fragment convert}}


# {{docs-fragment train}}
@env.task
async def train_one_epoch(df: DataFrame, batch_size: int = 128, seed: int = 0) -> dict:
    """Stream one shuffled epoch by random access. Nothing is downloaded whole."""
    ds = await df.open(lance.LanceDataset).all()

    order = list(range(ds.count_rows()))
    random.Random(seed).shuffle(order)

    seen = 0
    label_counts: dict[int, int] = {}
    for i in range(0, len(order), batch_size):
        # Reads only these rows, only these columns. Memory stays proportional to
        # the batch, not to the dataset.
        batch = ds.take(order[i : i + batch_size], columns=["image", "label"])
        for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            _ = len(image)  # stand-in for decoding and augmenting the image
            label_counts[label] = label_counts.get(label, 0) + 1
            seen += 1

    return {
        "rows_streamed": seen,
        # Map keys are stringified so the run UI renders them as text.
        "labels": {str(k): v for k, v in sorted(label_counts.items())},
    }
# {{/docs-fragment train}}


# {{docs-fragment metadata-scan}}
@env.task
async def label_histogram(df: DataFrame) -> dict:
    """Scan the structured columns only. The image bytes are never read."""
    ds = await df.open(lance.LanceDataset).all()

    counts: dict[int, int] = {}
    for batch in ds.scanner(columns=["label"], batch_size=4_096).to_batches():
        for label in batch.column("label").to_pylist():
            counts[label] = counts.get(label, 0) + 1
    return {str(k): v for k, v in sorted(counts.items())}
# {{/docs-fragment metadata-scan}}


# {{docs-fragment blob-file}}
@env.task
async def inspect_large_images(df: DataFrame, top_k: int = 3) -> list[int]:
    """Open blobs as file-like objects instead of loading them into memory."""
    ds = await df.open(lance.LanceDataset).all()

    sizes = []
    for blob in ds.take_blobs("image", indices=list(range(top_k))):
        with blob as f:
            sizes.append(len(f.readall()))
    return sizes
# {{/docs-fragment blob-file}}


# {{docs-fragment main}}
@env.task
async def main(n: int = 4_000) -> dict:
    dataset = await convert(n)
    return {
        "epoch": await train_one_epoch(dataset),
        "labels": await label_histogram(dataset),
        "blob_sizes": await inspect_large_images(dataset),
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment main}}
