import os
from typing import Annotated, Optional

import union

from .utils import EmbeddingConfig, VectorDB, VectorStoreConfig

image = union.ImageSpec(
    name="milvus-ingestion",
    packages=[
        "llama-index==0.12.28",
        "llama-index-vector-stores-milvus==0.7.2",
        "llama-index-embeddings-huggingface==0.5.2",
        "union==0.1.154",
    ],
)


@union.task(
    container_image=image,
    requests=union.Resources(mem="5Gi"),
    cache=union.Cache(ignored_inputs=["vector_db"], version="1"),
)
def ingest_docs(
    file_path: union.FlyteFile,
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
    vector_db: Optional[union.FlyteFile] = VectorDB.query(),
) -> Annotated[union.FlyteFile, VectorDB]:
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

    if not vector_db:
        milvus_vector_db = os.path.join(
            union.current_context().working_directory, "milvus_db.db"
        )
    else:
        milvus_vector_db = vector_db.download()

    vector_store = MilvusVectorStore(
        uri=milvus_vector_db,
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

    return union.FlyteFile(milvus_vector_db)


@union.workflow
def ingest_docs_workflow(
    file_path: union.FlyteFile,
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
    vector_db: Optional[union.FlyteFile] = VectorDB.query(),
) -> union.FlyteFile:
    return ingest_docs(
        file_path=file_path,
        vector_store_config=vector_store_config,
        embedding_config=embedding_config,
        vector_db=vector_db,
    )
