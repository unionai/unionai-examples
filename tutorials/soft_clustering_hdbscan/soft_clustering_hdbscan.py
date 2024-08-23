# # HDBSCAN Soft Clustering With Headline Embeddings on GPUs
#
# HDBSCAN is a state-of-the-art, density-based clustering algorithm that is used to
# uncover hidden patterns and structures in data. Some common applications of HDBSCAN
# include custom segmentation, anomaly detection, document clustering, and bio-informatics.
# In this tutorial, we will use the NVIDIA RAPIDS `cuML`'s version of HDBSCAN and UMAP
# to find soft clusters in a headlines dataset. We'll configure Flyte tasks to use
# NVIDIA's `A100` accelerators to embed the dataset and RAPIDS `cuML` for clustering.

# ## Downloading data
#
# We start by importing the dependencies for this workflow. Then, we download
# the headline dataset and cache it using `cache=True` and a `cache_version`:

import os
import tarfile
from pathlib import Path
from typing import Tuple

from flytekit import task, workflow, Resources, ImageSpec, current_context, Deck
from flytekit.deck import DeckField
from flytekit.types.file import FlyteFile
from flytekit.extras.accelerators import A100
import fsspec


@task(requests=Resources(cpu="2", mem="2Gi"), cache=True, cache_version="v1")
def download_headline_data() -> FlyteFile:
    headline_data = (
        "https://github.com/thomasjpfan/headlines-data/raw/main/headlines.parquet"
    )
    new_file = FlyteFile.new_remote_file("headlines.parquet")

    with fsspec.open(headline_data, "rb") as r:
        with new_file.open("wb") as w:
            w.write(r.read())

    return new_file


# ## Defining Python Dependencies
#
# The tasks in this workflow require python dependencies such as RAPIDS' `cuML` for
# clustering and `sentence_transformers` for embedding our headline dataset. Here we use
# `flytekit`'s `ImageSpec` to build an image with our require dependencies.

image = ImageSpec(
    name="sentence-transformer",
    python_version="3.11",
    packages=["union", "sentence-transformers==3.0.1"],
    conda_packages=[
        "cuml=24.08",
        "scikit-learn==1.4.*",
        "pytorch-cuda=12.1",
        "pytorch==2.4.0",
    ],
    conda_channels=["nvidia", "pytorch", "rapidsai"],
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)


# ## Embedding the headline data
#
# For embedding the headlines, we will use the
# [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
# model to convert the headlines into a 384 dimensional vector. For faster iterations,
# we define a `download_sentence_transformer` task to download the model and cache
# it with Union.


@task(
    requests=Resources(cpu="2", mem="6Gi"),
    container_image=image,
    cache=True,
    cache_version="v2",
)
def download_sentence_transformer() -> FlyteFile:
    from sentence_transformers import SentenceTransformer

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    model_cache = working_dir / "sentence_model"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(os.fspath(model_cache))

    model_cache_compressed = working_dir / "sentence_model.tar.gz"
    _compress(model_cache, model_cache_compressed)

    return model_cache_compressed


# Finally, we use the sentence transformer to embed the headline data into an embedding
# matrix with 384 columns and the number of rows is equal to the number of headlines.
# With `accelerator=A100` and `gpu="1"`, the `SentenceTransformer` uses the GPU to
# compute the embedding.


@task(
    requests=Resources(gpu="1", cpu="2", mem="2Gi"),
    accelerator=A100,
    container_image=image,
    cache=True,
    cache_version="v1",
)
def embed_headlines(
    headline_data: FlyteFile, sentence_transformer: FlyteFile
) -> FlyteFile:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pandas as pd

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    sentence_model = working_dir / "sentence_transformer"
    sentence_model.mkdir(exist_ok=True)

    # Load headline data
    df = pd.read_parquet(headline_data.remote_source)

    # Load sentence transformer
    _decompress(Path(sentence_transformer), sentence_model)

    model = SentenceTransformer(os.fspath(sentence_model), local_files_only=True)
    embeddings = model.encode(df["headline_text"])

    # Serialize model
    embedding_path = working_dir / "embedding.npy"

    np.save(embedding_path, embeddings)
    return embedding_path


# ## Soft Clustering
#
# Next, we use UMAP from RAPIDS `cuML` to reduce the dimensionality of the headline
# embeddings and pipe the results into HDBSCAN to soft cluster the data. Given that
# `cuML`'s UMAP and HDBSCAN are GPU accelerated, we set `accelerator=A100` to
# run the task with a GPU.


@task(
    requests=Resources(gpu="1", cpu="2", mem="2Gi"),
    accelerator=A100,
    container_image=image,
    cache=True,
    cache_version="v1",
)
def soft_clustering(embeddings: FlyteFile) -> Tuple[FlyteFile, FlyteFile]:
    # _configure_nvidia_libs()
    home_dir = Path("/") / "home" / ".lib"
    for p in home_dir.rglob("*"):
        print(p)
    import numpy as np
    import cuml

    embeddings.download()
    embeddings_np = np.load(embeddings.path)

    umap = cuml.manifold.UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0, random_state=12
    )
    reduced_data = umap.fit_transform(embeddings_np)

    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=50, metric="euclidean", prediction_data=True
    )
    clusterer.fit(reduced_data)
    soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)

    # Save clusters
    ctx = current_context()
    working_dir = Path(ctx.working_directory)

    cluster_labels_path = working_dir / "cluster_labels.npy"
    soft_cluster_path = working_dir / "soft_clusters.npy"

    np.save(cluster_labels_path, clusterer.labels_)
    np.save(soft_cluster_path, soft_clusters)
    return cluster_labels_path, soft_cluster_path


# ## Plotting Cluster Membership Uncertainty
#
# For plotting the soft clustering results, we define another `ImageSpec` that contains
# the requirements for plotting.

plot_image = ImageSpec(
    name="plot_cluster",
    packages=["numpy==1.26.4", "union", "seaborn==0.13.2", "matplotlib==3.9.1"],
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)


# Soft clustering assigns probabilities to each point that represents the likelihood a
# headline belongs to a cluster. To measure the confidence of each point, we take the
# difference between the probabilities between the point's top two clusters.
# In this Flyte task, we set `enable_deck=True` and build a histogram and
# empirical cumulative distribution to visualize the probability differences.


@task(
    requests=Resources(cpu="2", mem="2Gi"),
    container_image=plot_image,
    enable_deck=True,
    deck_fields=[DeckField.SOURCE_CODE, DeckField.DEPENDENCIES],
)
def plot_cluster_membership_uncertainty(
    cluster_labels: FlyteFile, soft_clusters: FlyteFile
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    cluster_labels.download()
    soft_clusters.download()

    cluster_labels_np = np.load(cluster_labels.path)
    soft_clusters_np = np.load(soft_clusters.path)

    soft_non_noise = soft_clusters_np[cluster_labels_np != -1]
    probs_top2_non_noise = np.take_along_axis(
        soft_non_noise, soft_non_noise.argsort(), axis=1
    )[:, -2:]
    diffs = np.diff(probs_top2_non_noise).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)
    fig.suptitle("Cluster Membership Uncertainty Evaluation")

    sns.histplot(ax=axes[0], data=diffs)
    axes[0].set_title("Difference between top two membership probabilities")

    sns.ecdfplot(ax=axes[1], data=diffs)
    axes[1].set_title("Cumulative distribution of differences")

    ctx = current_context()
    cluster_deck = Deck("Cluster Membership", _fig_to_html(fig))
    ctx.decks.insert(0, cluster_deck)


# ## Workflow


# Finally, we define the workflow that calls each Flyte task and route the data between
# each task. We run the workflow with:
#
# ```bash
# union run --remote soft_clustering_hdbscan.py hdscan_wf
# ````


@workflow
def hdscan_wf():
    headline_data = download_headline_data()
    sentence_transformer = download_sentence_transformer()
    embeddings = embed_headlines(
        headline_data=headline_data, sentence_transformer=sentence_transformer
    )

    cluster_labels, soft_cluster_path = soft_clustering(embeddings=embeddings)
    plot_cluster_membership_uncertainty(
        cluster_labels=cluster_labels, soft_clusters=soft_cluster_path
    )


# ## Appendix
#
# The following are helper functions used by our Flyte tasks. We include functions that
# decompress & compress tar files, and convert a matplotlib figure into HTML.


def _compress(src: Path, dest: Path):
    """Compress src into a tarfile."""
    import tarfile

    with tarfile.open(dest, "w:gz") as tar:
        for file in src.rglob("*"):
            tar.add(file, arcname=file.relative_to(src))


def _decompress(src: Path, dest: Path):
    """Decompress a tarfile into dest."""
    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(path=dest)


def _fig_to_html(fig) -> str:
    """Convert matplotlib figure to HTML."""
    import io
    import base64

    fig_bytes = io.BytesIO()
    fig.savefig(fig_bytes, format="jpg")
    fig_bytes.seek(0)
    image_base64 = base64.b64encode(fig_bytes.read()).decode()
    return f'<img src="data:image/png;base64,{image_base64}" alt="Rendered Image" />'
