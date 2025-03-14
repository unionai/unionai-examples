import os

import union
from flytekit.exceptions.base import FlyteRecoverableException

from .frontend.utils import (
    EmbeddingConfig,
    TextSplitterConfig,
    VectorStoreConfig,
    get_vector_store,
)

image = union.ImageSpec(
    name="enterprise-rag-ingestion",
    packages=[
        "langchain==0.3.19",
        "unstructured[all-docs]==0.16.23",
        "langchain-unstructured==0.1.6",
        "langchain-milvus==0.1.8",
        "langchain-huggingface==0.1.2",
        "langchain-nvidia-ai-endpoints==0.3.9",
    ],
    apt_packages=["libgl1", "libglib2.0-0"],
    registry=os.getenv("REGISTRY"),
)


@union.task(
    retries=3,
    secret_requests=[
        union.Secret(
            key="milvus-uri",
            env_var="MILVUS_URI",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
        union.Secret(
            key="milvus-token",
            env_var="MILVUS_TOKEN",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
    container_image=image,
    requests=union.Resources(mem="5Gi"),
    cache=True,
)
def ingest_docs(
    file_path: union.FlyteFile,
    text_splitter_config: TextSplitterConfig = TextSplitterConfig(),
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
) -> union.FlyteFile:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_unstructured import UnstructuredLoader

    local_file_path = file_path.download()

    not_supported_formats = (".rst", ".rtf", ".org")
    if local_file_path.endswith(not_supported_formats):
        raise ValueError(f"File format for {local_file_path} is not supported.")

    try:
        raw_documents = UnstructuredLoader(local_file_path).load()

        # Remove "languages" from metadata
        for doc in raw_documents:
            if "languages" in doc.metadata:
                del doc.metadata["languages"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_config.chunk_size,
            chunk_overlap=text_splitter_config.chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)
        vector_store = get_vector_store(
            os.getenv("MILVUS_URI"),
            os.getenv("MILVUS_TOKEN"),
            vector_store_config,
            embedding_config,
        )

        vector_store.add_documents(documents)
        return file_path
    except Exception:
        raise FlyteRecoverableException(
            "Connection timed out while making a request to the embedding model endpoint. Verify if the server is available."
        )
