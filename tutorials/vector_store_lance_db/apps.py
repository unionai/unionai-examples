import union
from union.app import App, Input


VectorStore = union.Artifact(name="ArxivPaperVectorStore")


image = union.ImageSpec(
    name="arxiv-rag-app-image",
    packages=[
        "lancedb",
        "pyarrow",
        "sentence-transformers",
        "google-genai",
        "fastapi[standard]",
        "union-runtime",
    ],
)


fastapi_app = App(
    name="arxiv-rag-app",
    container_image=image,
    inputs=[
        Input(
            value=VectorStore.query(),
            download=True,
            env_var="VECTOR_STORE_PATH",
        ),
    ],
    secrets=[union.Secret(name="google_api_key", env_var="GOOGLE_API_KEY")],
    limits=union.Resources(cpu="1", mem="2Gi", ephemeral_storage="4Gi"),
    port=8082,
    include=["fastapi_app.py"],
    args="fastapi dev --port 8082",
)
