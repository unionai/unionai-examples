import os

import union

from utils import EmbeddingConfig, VectorStoreConfig

image = union.ImageSpec(
    name="milvus-ingestion",
    packages=[
        "llama-index==0.12.28",
        "llama-index-vector-stores-milvus==0.7.2",
        "llama-index-embeddings-huggingface==0.5.2",
        "union==0.1.154",
    ],
    builder="union",
)


@union.task(
    container_image=image,
    requests=union.Resources(mem="5Gi"),
    cache=True,
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
)
def ingest_docs(
    file_path: union.FlyteFile,
    vector_store_config: VectorStoreConfig,
    embedding_config: EmbeddingConfig,
) -> str:
    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.milvus import MilvusVectorStore

    local_file_path = file_path.download()
    not_supported_formats = (".rst", ".rtf", ".org")
    if local_file_path.endswith(not_supported_formats):
        raise ValueError(f"File format for {local_file_path} is not supported.")

    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_config.model_name)

    reader = SimpleDirectoryReader(input_files=[local_file_path])
    documents = reader.load_data()

    vector_store = MilvusVectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name=vector_store_config.collection_name,
        dim=embedding_config.dimensions,
        index_config={
            "index_type": vector_store_config.index_type,
            "metric_type": "L2",
            "params": {"nlist": vector_store_config.nlist},
        },
        search_config={"nprobe": vector_store_config.nprobe},
        overwrite=False,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return os.path.basename(local_file_path)


@union.workflow
def ingest_docs_workflow(
    file_path: union.FlyteFile,
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
) -> str:
    return ingest_docs(
        file_path=file_path,
        vector_store_config=vector_store_config,
        embedding_config=embedding_config,
    )
