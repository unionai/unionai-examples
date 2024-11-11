# # Generating Wikipedia Embeddings
#
# In this tutorial, we’ll cover how to create embeddings for the Wikipedia dataset, powered by Union actors.
# The real advantage here is efficiency: with Union actors, you can load the model onto the GPU once and
# keep it ready for generating embeddings without reloading, which makes the process both fast and scalable.
#
# In short, by setting up an actor task, you can elevate a standard batch job to near-real-time inference.
#
# Let’s get started by importing the necessary libraries and modules:

import logging
import pathlib
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from flytekit import (
    ImageSpec,
    Resources,
    Secret,
    StructuredDataset,
    current_context,
    dynamic,
    task,
    workflow,
)
from flytekit.core.utils import timeit
from flytekit.types.directory import FlyteDirectory
from sentence_transformers import SentenceTransformer
from union.actor import ActorEnvironment

# ## Creating a secret
#
# Go to the HuggingFace website to generate an API key.
# Use the union CLI tool to create a secret:
#
# ```bash
# union create secret hf-api-key
# ```
#
# When prompted, paste the API key.

SERVERLESS_HF_KEY = "hf-api-key"

# ## Defining the imagespec and actor environment
#
# Define the container image specification, including all necessary dependencies,
# along with the actor environment setup.

embedding_image = ImageSpec(
    name="wikipedia_embedder",
    packages=[
        "datasets",
        "sentence_transformers",
        "pandas",
        "flytekit>=1.13.9",
        "requests>=2.29.0",
        "union>=0.1.86",
    ],
    python_version="3.11",
)

actor = ActorEnvironment(
    name="wikipedia-embedder-env",
    replica_count=20,
    ttl_seconds=900,
    requests=Resources(gpu="1", mem="12Gi", cpu="5"),
    container_image=embedding_image,
)

# We assign a GPU to the actor to run encoding on accelerated compute.
# Setting replicas to 20 provisions 20 workers to run the tasks concurrently.
#
# ## Caching the model
#
# To avoid downloading the model for each execution,
# we cache the model download task, ensuring the model remains available in the cache.

@task(
    cache=True,
    cache_version="0.1",
    requests=Resources(mem="5Gi"),
    secret_requests=[Secret(key=SERVERLESS_HF_KEY)],
    container_image=embedding_image,
)
def download_model(embedding_model: str) -> FlyteDirectory:
    from huggingface_hub import login, snapshot_download

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    login(token=ctx.secrets.get(key=SERVERLESS_HF_KEY))
    snapshot_download(embedding_model, local_dir=cached_model_dir)
    return FlyteDirectory(path=cached_model_dir)

# ## Defining an actor task
#
# Define an actor task to handle data encoding.
# The encoder is set as a global variable to prevent re-initialization with each encoding task.

encoder: Optional[SentenceTransformer] = None


def load_model(local_model_path: str) -> SentenceTransformer:
    global encoder
    if encoder:
        return encoder
    encoder = SentenceTransformer(local_model_path)
    encoder.max_seq_length = 256
    return encoder


@actor.task(cache=True, cache_version="1.1")
def encode(
    df: pd.DataFrame, batch_size: int, model_dir: FlyteDirectory
) -> torch.Tensor:
    bi_encoder = load_model(model_dir.download())
    print(f"Loaded encoder into device: {bi_encoder.device}")
    return bi_encoder.encode(
        df["text"],
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

# ## Defining a partitions task
#
# Next, define a task to list all partitions for the dataset.
# This task downloads the data and iterates over the directory to create partitions.

@task(
    container_image=embedding_image,
    requests=Resources(mem="4Gi", cpu="3"),
    cache=True,
    cache_version="1.1",
)
def list_partitions(name: str, version: str, num_proc: int) -> list[StructuredDataset]:
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
            partitions.append(StructuredDataset(uri=str(f)))
        i += 1
    return partitions

# ## Defining workflows
#
# Define a dynamic workflow that loops through the partitions and calls the encode task to generate embeddings. 
# After the first run, subsequent encode tasks reuse the actor environment, leading to faster encoding.

@dynamic(
    container_image=embedding_image,
    cache=True,
    cache_version="1.1",
)
def dynamic_encoder(
    partitions: list[StructuredDataset], model_dir: FlyteDirectory, batch_size: int
) -> list[torch.Tensor]:
    embeddings = []
    for p in partitions:
        embeddings.append(encode(df=p, batch_size=batch_size, model_dir=model_dir))
    return embeddings

# Next, define a workflow that sequentially calls all tasks, including the dynamic workflow. 
# The output will be a list of embeddings in the form of Torch tensors.

@workflow
def embed_wikipedia(
    name: str = "wikipedia",
    version: str = "20220301.en",
    num_proc: int = 4,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 128,
) -> list[torch.Tensor]:
    partitions = list_partitions(name=name, version=version, num_proc=num_proc)
    model_dir = download_model(embedding_model=embedding_model)
    return dynamic_encoder(
        partitions=partitions, model_dir=model_dir, batch_size=batch_size
    )
