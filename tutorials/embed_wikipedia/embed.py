import functools
import logging
import pathlib
import typing

import pandas as pd
import torch
from datasets import load_dataset_builder, DownloadConfig
from flytekit import ImageSpec, task, Resources, workflow, dynamic, StructuredDataset
from flytekit.core.utils import timeit
from sentence_transformers import SentenceTransformer
from union.actor import ActorEnvironment

embedding_image = ImageSpec(
    # requirements="requirements.txt",
    packages=["datasets", "sentence_transformers", "pandas", "union>=0.1.51", "requests>=2.29.0"],
    registry="ghcr.io/unionai-oss",
    python_version="3.11",
)

actor = ActorEnvironment(
    name="embedding-actor",
    replica_count=2 * 8,
    parallelism=1,
    backlog_length=0,
    ttl_seconds=300,
    requests=Resources(gpu="1", mem="12Gi", cpu="7"),
    # accelerator=A10G,
    container_image=embedding_image,
    # secrets=secret_requests=[Secret(key="hf_token")],
)

encoder: typing.Optional[SentenceTransformer] = None


@timeit("load_model")
def load_model(model_name: str = 'msmarco-MiniLM-L-6-v3') -> SentenceTransformer:
    global encoder
    if encoder:
        return encoder
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = 256
    return encoder


@actor.task(cache=True, cache_version="1.8")
def encode(model: str, df: pd.DataFrame, batch_size: int) -> torch.Tensor:
    bi_encoder = load_model(model)
    print(f"Loaded encoder into device: {bi_encoder.device}")
    with timeit("encode"):
        return bi_encoder.encode(df["text"], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)


@task(container_image=embedding_image, requests=Resources(mem="4Gi", cpu="3"), retries=0, cache=True,
            cache_version="1.0")
def list_partitions(name: str, version: str, num_proc: int) -> typing.List[StructuredDataset]:
    dsb = load_dataset_builder(name, version, cache_dir="/tmp/hfds", trust_remote_code=True)
    logging.log(logging.INFO, f"Downloading {name} {version}")
    with timeit("download"):
        dsb.download_and_prepare(file_format="parquet",
                                 download_config=DownloadConfig(disable_tqdm=True, num_proc=num_proc))
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


@dynamic(container_image=embedding_image, cache=True, cache_version="2.3")
def dynamic_encoder(partitions: typing.List[StructuredDataset], embedding_model: str, batch_size: int) -> typing.List[
    torch.Tensor]:
    embeddings = []
    for p in partitions:
        embeddings.append(encode(model=embedding_model, df=p, batch_size=batch_size))  # noqa
    return embeddings


@workflow
def embed_wikipedia(
        name: str = "wikipedia",
        version: str = "20220301.simple",
        num_proc: int = 4,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 128,
) -> typing.List[torch.Tensor]:
    partitions = list_partitions(name=name, version=version, num_proc=num_proc)
    return dynamic_encoder(partitions=partitions, embedding_model=embedding_model, batch_size=batch_size)  # noqa


if __name__ == '__main__':
    df = pd.read_parquet("/Users/ketanumare/src/data/wiki-2.parquet")
    s = df["text"][:1000]
    new_df = pd.DataFrame(s)
    m = 'msmarco-MiniLM-L-6-v3'
    m = "BAAI/bge-small-en-v1.5"
    encode(model=m, df=new_df, batch_size=64)
