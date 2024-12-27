# # Generating Wikipedia Embeddings
#
# This tutorial demonstrates how to create embeddings for the Wikipedia dataset using Union actors.
# Union actors facilitate efficient processing by keeping the model readily available for generating embeddings,
# eliminating the need for repeated downloads.
#
# In short, by setting up an actor task, you can elevate a standard batch job to near-real-time inference.
#
# Let’s get started by importing the necessary libraries and modules:

import functools
import logging
import pathlib
import shutil
from pathlib import Path

import flytekit as fl
import pandas as pd
import torch
from flytekit.core.utils import timeit
from flytekit.types.directory import FlyteDirectory
from union.actor import ActorEnvironment

# ## Creating a secret
#
# This workflow requires a HuggingFace API key. To set up the secret:
#
# 1. Generate an API key from the HuggingFace website
# 2. Create a secret using the Union CLI:
#
# ```bash
# union create secret hf-api-key
# ```

SERVERLESS_HF_KEY = "hf-api-key"

# ## Defining the imagespec and actor environment
#
# The container image specification includes all required dependencies,
# while the actor environment defines the runtime parameters for execution.

embedding_image = fl.ImageSpec(
    name="wikipedia_embedder",
    packages=[
        "datasets",
        "sentence_transformers",
        "pandas",
        "requests>=2.29.0",
        "union>=0.1.117",
    ],
    python_version="3.11",
)

actor = ActorEnvironment(
    name="wikipedia-embedder-env",
    replica_count=20,
    ttl_seconds=900,
    requests=fl.Resources(gpu="1", mem="12Gi", cpu="5"),
    container_image=embedding_image,
)

# The actor configuration assigns a GPU for accelerated encoding and
# provisions 20 workers for concurrent task execution.
#
# ## Caching the model
#
# We cache the model download task to ensure the model remains available
# between executions, avoiding redundant downloads from the HuggingFace hub.


@fl.task(
    cache=True,
    cache_version="0.1",
    requests=fl.Resources(mem="5Gi"),
    secret_requests=[fl.Secret(key=SERVERLESS_HF_KEY)],
    container_image=embedding_image,
)
def download_model(embedding_model: str) -> FlyteDirectory:
    from huggingface_hub import login, snapshot_download

    ctx = fl.current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    login(token=ctx.secrets.get(key=SERVERLESS_HF_KEY))
    snapshot_download(embedding_model, local_dir=cached_model_dir)
    return FlyteDirectory(path=cached_model_dir)


# ## Defining an actor task
#
# An actor task is used to handle data encoding efficiently.
# To optimize performance, the model is not reloaded if it is already available locally.
# Actor tasks enable this by allowing you to reuse a container across multiple tasks.


@actor.task(cache=True, cache_version="1.1")
def encode(
    df: pd.DataFrame, batch_size: int, model_dir: FlyteDirectory
) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    local_path = Path("/tmp/embedding-model")

    if not local_path.exists():
        temp_local_path = model_dir.download()
        shutil.copytree(src=temp_local_path, dst=str(local_path))

    encoder = SentenceTransformer(str(local_path))
    encoder.max_seq_length = 256

    print(f"Loaded encoder into device: {encoder.device}")
    return encoder.encode(
        df["text"],
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )


# ## Defining a partitions task
#
# Next, we define a task to list all partitions for the dataset.
# This task downloads the data and iterates over the directory to create partitions.


@fl.task(
    container_image=embedding_image,
    requests=fl.Resources(mem="4Gi", cpu="3"),
    cache=True,
    cache_version="1.1",
)
def list_partitions(
    name: str, version: str, num_proc: int
) -> list[fl.StructuredDataset]:
    from datasets import DownloadConfig, load_dataset_builder

    dsb = load_dataset_builder(
        name, version, cache_dir="/tmp/hfds", trust_remote_code=True
    )
    logging.log(logging.INFO, f"Downloading {name} {version}")
    with timeit("download"):
        dsb.download_and_prepare(
            file_format="parquet",
            download_config=DownloadConfig(disable_tqdm=True, num_proc=num_proc),
        )
    logging.log(logging.INFO, f"Download complete")
    p = pathlib.Path(dsb.cache_dir)
    i = 0
    partitions = []
    for f in p.iterdir():
        if "parquet" in f.name:
            print(f"Encoding {i}: {f}")
            partitions.append(fl.StructuredDataset(uri=str(f)))
        i += 1
    return partitions


# ## Defining a workflow
#
# We define a workflow that sequentially calls all tasks, including the map task,
# which iterates through the partitions and invokes the encode task to generate embeddings.
# After the initial run, subsequent encode tasks reuse the actor environment, resulting in faster encoding.
# This workflow returns a list of embeddings as Torch tensors.


@fl.workflow
def embed_wikipedia(
    name: str = "wikipedia",
    version: str = "20220301.en",
    num_proc: int = 4,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 128,
) -> list[torch.Tensor]:
    partitions = list_partitions(name=name, version=version, num_proc=num_proc)
    model_dir = download_model(embedding_model=embedding_model)
    return fl.map_task(
        functools.partial(encode, batch_size=batch_size, model_dir=model_dir)
    )(df=partitions)
