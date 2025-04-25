from contextlib import asynccontextmanager

import flytekit as fl
import union
from fastapi import FastAPI
from flytekit import Resources
from flytekit.extras.accelerators import A100
from union.app import App, Input
from union.app.llm import VLLMApp
from utils import EmbeddingConfig, VectorDB

LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

llm_image = union.ImageSpec(
    name="vllm-deepseek",
    packages=["union-runtime>=0.1.17", "vllm==0.8.4"],
)

deepseek_app = VLLMApp(
    name="vllm-deepseek",
    container_image=llm_image,
    limits=Resources(cpu="2", mem="20Gi", gpu="1", ephemeral_storage="20Gi"),
    accelerator=A100,
    requires_auth=False,
    model_id="deepseek-qwen-1.5b",
    scaledown_after=300,
    stream_model=True,
)

#########################
# PHOENIX MODEL TRACING #
#########################
gradio_app = App(
    name="vllm-deepseek-gradio-phoenix",
    inputs=[
        Input(
            name="vllm_deepseek_endpoint",
            value=deepseek_app.endpoint,
            env_var="VLLM_DEEPSEEK_ENDPOINT",
        ),
        Input(
            name="phoenix_endpoint",
            value="https://app.phoenix.arize.com",
            env_var="ENDPOINT",
        ),
    ],
    container_image=union.ImageSpec(
        name="vllm-deepseek-gradio",
        packages=[
            "gradio==5.23.3",
            "union-runtime>=0.1.17",
            "openinference-instrumentation-openai==0.1.23",
            "opentelemetry-semantic-conventions==0.52b1",
            "openai==1.71.0",
            "arize-phoenix==8.22.1",
        ],
    ),
    limits=union.Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["gradio_app.py"],
    args=[
        "python",
        "gradio_app.py",
    ],
    secrets=[
        fl.Secret(
            key="phoenix-api-key",
            env_var="PHOENIX_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
    dependencies=[deepseek_app],
)


#####################
# ARIZE RAG TRACING #
#####################
@asynccontextmanager
async def lifespan(app):
    import os

    from arize.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor

    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        api_key=os.getenv("ARIZE_API_KEY"),
        project_name="arize-union",
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    yield


fastapi_app = FastAPI(lifespan=lifespan)

arize_app = App(
    name="rag-fastapi-arize",
    inputs=[
        Input(
            name="vector_db",
            value=VectorDB.query(),
            download=True,
            mount="/root/vector_db",
        ),
        Input(name="arize-space-id", value="default", env_var="ARIZE_SPACE_ID"),
    ],
    containter_image=union.ImageSpec(
        name="rag-fastapi-arize",
        packages=[
            "llama-index==0.12.28",
            "llama-index-embeddings-huggingface==0.5.2",
            "llama-index-vector-stores-milvus==0.7.2",
            "llama-index-llms-openai==0.3.30",
            "union-runtime>=0.1.17",
        ],
    ),
    secrets=[
        fl.Secret(
            key="arize-api-key",
            env_var="ARIZE_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
    framework_app=fastapi_app,
    dependencies=[deepseek_app],
)


@arize_app.get("/query_rag")
async def query_rag(prompt: str) -> str:
    from llama_index.core import Settings, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        base_url=f"{deepseek_app.endpoint}/v1",
        api_key="abc",
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EmbeddingConfig.model_name)

    storage_context = StorageContext.from_defaults(persist_dir="/root/vector_db")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()

    response = query_engine.query(prompt)
    return response
