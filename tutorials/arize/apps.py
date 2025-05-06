import os
from contextlib import asynccontextmanager

import union
from fastapi import FastAPI
from flytekit.extras.accelerators import L4
from union import Resources
from union.app import App, ArizeConfig, Input, PhoenixConfig
from union.app.llm import VLLMApp
from utils import EmbeddingConfig, VectorStoreConfig

LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ID = "deepseek-qwen-1.5b"


llm_image = union.ImageSpec(
    name="vllm-deepseek",
    packages=[
        "union[vllm]==0.1.181",
    ],
    apt_packages=["build-essential"],
    builder="union",
)

deepseek_app = VLLMApp(
    name="vllm-deepseek",
    container_image=llm_image,
    # TODO: add the model artifact uri
    model="<YOUR_MODEL_ARTIFACT_URI>",
    model_id=MODEL_ID,
    scaledown_after=1200,
    stream_model=True,
    limits=Resources(mem="23Gi", gpu="1", ephemeral_storage="20Gi", cpu="6"),
    accelerator=L4,
    requires_auth=False,
)


#########################
# PHOENIX MODEL TRACING #
#########################
gradio_image = union.ImageSpec(
    name="vllm-deepseek-gradio",
    packages=[
        "gradio==5.23.3",
        "union-runtime>=0.1.17",
        "openinference-instrumentation-openai==0.1.23",
        "opentelemetry-semantic-conventions==0.52b1",
        "openai==1.71.0",
        "arize-phoenix==8.22.1",
    ],
    builder="union",
)

gradio_app = App(
    name="vllm-deepseek-gradio-phoenix",
    inputs=[
        Input(
            name="vllm_deepseek_endpoint",
            value=deepseek_app.query_endpoint(public=False),
            env_var="VLLM_DEEPSEEK_ENDPOINT",
        ),
    ],
    container_image=gradio_image,
    limits=union.Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["gradio_app.py"],
    args=[
        "python",
        "gradio_app.py",
    ],
    config=PhoenixConfig(
        endpoint="https://app.phoenix.arize.com", project="phoenix-union"
    ),
    secrets=[
        union.Secret(
            key="phoenix-api-key",
            mount_requirement=union.Secret.MountType.ENV_VAR,
            env_var="PHOENIX_API_KEY",
        )
    ],
    dependencies=[deepseek_app],
)


#####################
# ARIZE RAG TRACING #
#####################
arize_image = union.ImageSpec(
    name="rag-fastapi-arize",
    packages=[
        "llama-index==0.12.28",
        "llama-index-embeddings-huggingface==0.5.2",
        "llama-index-vector-stores-milvus==0.7.2",
        "llama-index-llms-openai-like==0.3.4",
        "union-runtime>=0.1.17",
        "arize-otel==0.8.1",
        "opentelemetry-exporter-otlp==1.32.1",
        "fastapi[standard]==0.115.12",
        "openinference-instrumentation-llama-index==4.2.1",
    ],
    builder="union",
)


@asynccontextmanager
async def lifespan(app):
    from arize.otel import register
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    tracer_provider = register()
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    yield


fastapi_app = FastAPI(lifespan=lifespan)
arize_app = App(
    name="rag-fastapi-arize",
    inputs=[
        Input(
            name="arize-space-id",
            value="default",
            env_var="ARIZE_SPACE_ID",
        ),
        Input(
            name="arize-project-name", value="arize-union", env_var="ARIZE_PROJECT_NAME"
        ),
        Input(
            name="vllm_deepseek_endpoint",
            value=deepseek_app.query_endpoint(public=False),
            env_var="VLLM_DEEPSEEK_ENDPOINT",
        ),
    ],
    container_image=arize_image,
    secrets=[
        union.Secret(
            key="arize-api-key",
            env_var="ARIZE_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
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
    config=ArizeConfig(
        endpoint="https://app.arize.com/organizations/<YOUR_ORG_ID>/spaces/<YOUR_SPACE_ID>/models/<YOUR_MODEL_ID>"  # TODO: Add your arize org, space, and model IDs.
    ),
    framework_app=fastapi_app,
    dependencies=[deepseek_app],
    limits=union.Resources(cpu="1", mem="5Gi"),
)


@fastapi_app.post("/query_rag")
async def query_rag(
    prompt: str,
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
) -> str:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.vector_stores.milvus import MilvusVectorStore

    Settings.llm = OpenAILike(
        model=MODEL_ID,
        api_base=f"{os.getenv("VLLM_DEEPSEEK_ENDPOINT")}/v1",
        api_key="abc",
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_config.model_name)

    milvus_store = MilvusVectorStore(
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

    index = VectorStoreIndex.from_vector_store(vector_store=milvus_store)
    query_engine = index.as_query_engine()

    response = query_engine.query(prompt)
    return response.response
