# # Taking NVIDIA’s Enterprise RAG Blueprint to Production
#
# This example builds on the `1.0.0` enterprise RAG NVIDIA blueprint and shows how to adapt it into a modular, production-ready deployment on Union.

# {{run-on-union}}

# ## Union’s role in this setup
#
# NVIDIA Blueprints are powerful Compound AI systems, but getting them into production usually requires a dedicated infrastructure/platform expert
# and multiple subject matter experts (SMEs) to deploy the full solution.

# Union makes this process easier by turning blueprints into enterprise-ready workflows that you can develop locally and deploy to production.
# It provides all the necessary components to build production-grade Compound AI systems, from data processing to model serving.

# When you convert the enterprise RAG Blueprint to a Union workflow, you get:

# 1. **Data ingestion as a background job**: Data ingestion is handled as a Union task, with built-in capabilities like:
#
#    - **Retries**: If the vector database is down for some reason, the task automatically retries instead of failing immediately.
#    - **Caching**: The task won’t rerun for the same file, so there’s no need to manually delete documents from the vector database
#      as suggested in the blueprint. It just returns the cached output.
#    - **Secrets management**: Easily assign secrets with Union’s built-in secret interface.
#    - **Simplified image management**: No need to write complex Dockerfiles; just use Union’s imagespec to define images.
#
#    We're using a *hosted* [Milvus](https://milvus.io/) vector database to ensure scalability and a production-grade setup.
#
# 2. **Model serving**: All NVIDIA NIM models can be served using Union. You can:
#
#    - Store models as Union artifacts (to cache the models) and serve them later.
#    - Set resource limits (GPU, CPU, memory), environment variables, secrets, and more.
#
# 3. **RAG application serving**
#
#    - Beyond just models, you can also serve full RAG applications with Union, so everything is managed under one framework.
#    - The RAG app tracks the progress of the background data ingestion job using UnionRemote and reports its status.
#    - Unlike the blueprint, which used nested invocations, our proposed pipeline is more straightforward. It consists of a data ingestion task,
#      FastAPI services, and a Gradio app, with no boilerplate code. Union keeps the code centralized, making development and maintenance more efficient.
#
# 4. **Development to production**
#
#    - Test the data ingestion task locally before deploying it.
#    - Run the FastAPI app locally before deploying it as a full application on Union.
#    - Moving from local development to production is straightforward.

# ## Architecture
#
# The RAG system is composed of three core components, each deployed as a standalone Union app:
#
# - **Embedding model**: Converts input text into dense vector representations.
# - **Reranker model**: Scores and ranks retrieved documents based on relevance.
# - **Language model**: Generates natural language responses grounded in the retrieved context.

# A preprocessing step ingests your source data and prepares it for retrieval. This includes:
#
# - Generating vector embeddings using the embedding model.
# - Storing these vectors in Milvus.
#
# ```python
# @union.task(retries=3, ...)
# def ingest_docs(...) -> union.FlyteFile:
#     ...
#     try:
#         raw_documents = UnstructuredLoader(local_file_path).load()
#
#         # Remove "languages" from metadata
#         for doc in raw_documents:
#             if "languages" in doc.metadata:
#                 del doc.metadata["languages"]
#
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=text_splitter_config.chunk_size,
#             chunk_overlap=text_splitter_config.chunk_overlap,
#         )
#         documents = text_splitter.split_documents(raw_documents)
#         vector_store = get_vector_store(
#             os.getenv("MILVUS_URI"),
#             os.getenv("MILVUS_TOKEN"),
#             vector_store_config,
#             embedding_config,
#         )
#
#         vector_store.add_documents(documents)
#         return file_path
#     except Exception:
#         raise FlyteRecoverableException(
#             "Connection timed out while making a request to the embedding model endpoint. Verify if the server is available."
#         )
# ```
#
# > [!NOTE]
# > You can find the ingestion logic in
# > [`ingestion.py`](https://github.com/unionai/unionai-examples/blob/main/tutorials/nvidia_blueprints/enterprise_rag/ingestion.py).

# In addition to the model-serving apps, a fourth Union app brings everything together by running the end-to-end RAG workflow and serving a custom frontend that interacts with all components.

import os

import union
from artifact import (
    LLM_MODEL,
    NIMEmbeddingModel,
    NIMRerankerModel,
    enterprise_rag_embedding_image,
    enterprise_rag_reranker_image,
)
from flytekit.extras.accelerators import A100, L4
from union.app import App, Input

# We first generate Union artifacts for all three components: the embedding model, the reranker, and the LLM.
#
# ```python
# @union.task(...)
# def create_model() -> FlyteDirectory:
#     ...
#     os.environ["LOCAL_NIM_CACHE"] = nim_cache
#     os.environ["NIM_CACHE_PATH"] = nim_cache
#     os.environ["NGC_HOME"] = os.path.join(nim_cache, "ngc", "hub")
#
#     run(["download-to-cache"])
#     return nim_cache
#
# @union.workflow
# def download_nim_models_to_cache() -> tuple[
#     Annotated[FlyteDirectory, NIMEmbeddingModel],
#     Annotated[FlyteDirectory, NIMRerankerModel],
#     Annotated[FlyteDirectory, NIMLLMModel],
# ]:
#     return (
#         create_model(),
#         create_model().with_overrides(
#             container_image=enterprise_rag_reranker_image.image_name(),
#             requests=union.Resources(
#                 cpu="2", mem="16Gi", ephemeral_storage="16Gi", gpu="1"
#             ),
#             accelerator=L4,
#         ),
#         ...
#     )
# ```

# > [!NOTE]
# > You can find the artifact generation code in
# > [`artifact.py`](https://github.com/unionai/unionai-examples/blob/main/tutorials/nvidia_blueprints/enterprise_rag/artifact.py).

# We define a `download_nim_models_to_cache` workflow to download each model to a local cache directory.
# A single task handles this download step, and we override the container image and resource requests for each model individually in the workflow.

# For the embedding and reranker models, we reuse the same container images that were used to generate their Union Artifacts.

# For the LLM, we define a separate container image that includes all required environment variables and dependencies to run the model.

enterprise_rag_llm_image = union.ImageSpec(
    name="enterprise-rag-llm",
    base_image=union.ImageSpec(
        name="enterprise-rag-llm",
        base_image=f"nvcr.io/nim/{LLM_MODEL}",
        builder="default",
        registry=os.getenv("REGISTRY"),
    ),
    apt_packages=["curl"],
    env={
        "PATH": "/opt/nim/llm/.venv/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:$LD_LIBRARY_PATH",
    },
    commands=[
        "curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py",
        "/opt/nim/llm/.venv/bin/python /tmp/get-pip.py",
        "/opt/nim/llm/.venv/bin/python -m pip install union==0.1.151 union-runtime==0.1.11",
    ],
)

# We define individual apps to serve the embedding model, reranker, and LLM.
# Each app loads its respective model and handles incoming requests directly.

# We configure the apps to use the model artifacts generated in the previous step as inputs.

embedding_model = App(
    name="enterprise-rag-embedding",
    inputs=[
        Input(
            name="enterprise-rag-embedding-model",
            value=NIMEmbeddingModel.query(),
            download=True,
            mount="/root/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_embedding_image.with_packages(
        "union-runtime==0.1.11"
    ),
    args="/opt/nim/start-server.sh",
    requests=union.Resources(cpu="3", mem="25Gi", ephemeral_storage="16Gi", gpu="1"),
    env={
        "NIM_SERVER_PORT": "8080",
        "NIM_CACHE_PATH": "/root/nim/.cache",
    },
    accelerator=L4,
)

reranker_model = App(
    name="enterprise-rag-reranker",
    inputs=[
        Input(
            name="enterprise-rag-reranker-model",
            value=NIMRerankerModel.query(),
            download=True,
            mount="/root/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_reranker_image.with_packages(
        "union-runtime==0.1.11"
    ),
    args="/opt/nim/start-server.sh",
    env={"NIM_SERVER_PORT": "8080", "NIM_CACHE_PATH": "/root/nim/.cache"},
    requests=union.Resources(cpu="3", mem="30Gi", ephemeral_storage="16Gi", gpu="1"),
    accelerator=L4,
)

llm_model = App(
    name="enterprise-rag-llm",
    inputs=[
        Input(
            name="enterprise-rag-llm-model",
            value=union.Artifact(name="llama-31-8b-instruct").query(),
            download=True,
            mount="/root/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_llm_image,
    args="/opt/nim/start-server.sh",
    requests=union.Resources(cpu="12", mem="60Gi", ephemeral_storage="40Gi", gpu="1"),
    env={
        "NIM_SERVER_PORT": "8080",
        "NIM_CACHE_PATH": "/root/nim/.cache",
        "NGC_HOME": "/root/nim/.cache/ngc/hub",
    },
    accelerator=A100,
)

# We define the main RAG app as the entry point for the entire system.
# This app integrates the embedding, reranker, and LLM components, and provides a user interface for interacting with the RAG workflow.

# The app uses FastAPI to handle incoming requests and Gradio to power the frontend.
# It also uses UnionRemote to track the status of the data ingestion job, allowing users to monitor the progress of the background task in real time.

enterprise_rag_app = App(
    name="enterprise-rag",
    container_image=union.ImageSpec(
        name="enterprise-rag", requirements="requirements.txt"
    ),
    inputs=[
        Input(
            name="enterprise-rag-embedding-endpoint",
            value=embedding_model.query_endpoint(public=False),
            env_var="EMBEDDING_ENDPOINT",
        ),
        Input(
            name="enterprise-rag-reranker-endpoint",
            value=reranker_model.query_endpoint(public=False),
            env_var="RERANKER_ENDPOINT",
        ),
        Input(
            name="enterprise-rag-llm",
            value=llm_model.query_endpoint(public=False),
            env_var="LLM_ENDPOINT",
        ),
        Input(name="llm-model", value=LLM_MODEL, env_var="LLM_MODEL"),
    ],
    limits=union.Resources(cpu="1", mem="1Gi"),
    port=8080,
    min_replicas=1,
    max_replicas=1,
    include=["./frontend/**"],
    args="uvicorn frontend.fastapi_app_server:app --port 8080",
    secrets=[
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
        union.Secret(
            key="union-api-key",
            env_var="UNION_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
)

# > [!NOTE]
# > Explore the complete RAG app code [here](https://github.com/unionai/unionai-examples/tree/main/tutorials/nvidia_blueprints/enterprise_rag).
