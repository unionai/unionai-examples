import csv
import gzip
import shutil
from pathlib import Path

import union

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

image = union.ImageSpec(
    name="airbnb-weaviate-ingestion",
    packages=[
        "llama-index==0.12.32",
        "llama-index-vector-stores-weaviate==1.3.1",
        "union==0.1.178",
        "llama-index-embeddings-fastembed==0.3.1",
    ],
)


@union.task(
    cache=True,
    secret_requests=[
        union.Secret(key="weaviate_url", env_var="WEAVIATE_URL"),
        union.Secret(key="weaviate_api_key", env_var="WEAVIATE_API_KEY"),
    ],
    requests=union.Resources(cpu=2, mem="10Gi"),
    container_image=image,
)
def ingest_data(inside_airbnb_listings_url: union.FlyteFile, index_name: str) -> str:
    import os

    import weaviate
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(timeout=Timeout(init=30)),
    )

    output_file_path = (
        Path(union.current_context().working_directory) / "airbnb_listings.csv"
    )
    with gzip.open(inside_airbnb_listings_url.download(), "rb") as f_in:
        with open(output_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    documents = []

    FIELDS = {
        "price": int,
        "review_scores_rating": float,
        "number_of_reviews": int,
        "host_is_superhost": str,
        "instant_bookable": str,
        "neighbourhood_cleansed": str,
        "neighbourhood": str,
    }

    with open(output_file_path, encoding="utf-8") as fp:
        csv_reader = csv.DictReader(fp, delimiter=",", quotechar='"')
        for row in csv_reader:
            metadata = {}
            for key, convert_fn in FIELDS.items():
                raw_val = row.get(key, "").strip()
                if raw_val:
                    if (
                        convert_fn != str
                    ):  # Only convert if the type is not already string
                        try:
                            raw_val = (
                                raw_val.replace("$", "") if key == "price" else raw_val
                            )
                            metadata[key] = convert_fn(raw_val)
                        except ValueError:
                            continue
                    else:
                        metadata[key] = raw_val

            documents.append(
                Document(
                    text="\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items()),
                    metadata=metadata,
                )
            )

    Settings.embed_model = FastEmbedEmbedding(model_name=EMBEDDING_MODEL)

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )

    client.close()

    return "Ingestion complete"
