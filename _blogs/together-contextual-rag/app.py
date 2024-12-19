import os

import flytekit as fl
from flytekit.core.artifact import Artifact
from union.app import App, Endpoint, Input

BM25Index = Artifact(name="bm25s-index")
ContextualChunksJSON = Artifact(name="contextual-chunks-json")


chroma_app = Endpoint(
    name="contextual-rag-chroma-db-app",
    container_image=fl.ImageSpec(
        name="contextual-rag-chroma-db",
        registry=os.getenv("REGISTRY"),
        packages=["union-runtime", "chromadb"],
    ),
    limits=fl.Resources(cpu="3", mem="5Gi"),
    port=8080,
    min_replicas=1,
    max_replicas=1,
    command=["chroma", "run", "--port", "8080"],
)


fastapi_app = Endpoint(
    name="contextual-rag-fastapi-app",
    inputs=[
        Input(
            name="bm25s_index",
            value=BM25Index.query(),
            auto_download=True,
            env_name="BM25S_INDEX",
        ),
        Input(
            name="contextual_chunks_json",
            value=ContextualChunksJSON.query(),
            auto_download=True,
            env_name="CONTEXTUAL_CHUNKS_JSON",
        ),
        Input(
            name="chroma_db_endpoint",
            # value=chroma_app.query_endpoint(public=False),
            value="http://contextual-rag-chroma-db-app.demo-development.svc.cluster.local",  # TODO: Remove when the fix is in.
            env_name="CHROMA_DB_ENDPOINT",
        ),
    ],
    container_image=fl.ImageSpec(
        name="contextual-rag-fastapi",
        registry=os.getenv("REGISTRY"),
        packages=[
            "together",
            "bm25s",
            "chromadb",
            "fastapi[standard]",
            "union-runtime",
        ],
    ),
    limits=fl.Resources(cpu="3", mem="10Gi"),
    port=8080,
    include=["./main.py"],
    command=["fastapi", "dev", "--port", "8080"],
    min_replicas=1,
    max_replicas=1,
)


gradio_app = App(
    name="contextual-rag-gradio-app",
    inputs=[
        Input(
            name="fastapi_endpoint",
            # value=fastapi_app.query_endpoint(public=False),
            value="http://contextual-rag-fastapi-app.demo-development.svc.cluster.local",  # TODO: Remove when the fix is in.
            env_name="FASTAPI_ENDPOINT",
        )
    ],
    container_image=fl.ImageSpec(
        name="contextual-rag-gradio",
        registry=os.getenv("REGISTRY"),
        packages=["gradio", "union-runtime"],
    ),
    limits=fl.Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["./gradio_app.py"],
    command=[
        "python",
        "gradio_app.py",
    ],
    min_replicas=1,
    max_replicas=1,
)
