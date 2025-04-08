from contextlib import asynccontextmanager

import flytekit as fl
import union
from flytekit import Resources
from flytekit.extras.accelerators import A100
from union.app import App, FastAPIApp, Input
from utils import EmbeddingConfig, VectorDB

deepseek_app = App(
    name="vllm-deepseek",
    container_image="docker.io/vllm/vllm-openai:latest",
    command=[],
    args=[
        "--model",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--trust-remote-code",
    ],
    port=8000,
    limits=Resources(cpu="2", mem="20Gi", gpu="1", ephemeral_storage="20Gi"),
    env={
        "DEBUG": "1",
        "LOG_LEVEL": "DEBUG",
    },
    accelerator=A100,
)

#########################
# PHOENIX MODEL TRACING #
#########################
gradio_app = App(
    name="vllm-deepseek-gradio-phoenix",
    inputs=[
        Input(
            name="vllm_deepseek_endpoint",
            value=deepseek_app.query_endpoint(public=False),
            env_var="VLLM_DEEPSEEK_ENDPOINT",
        ),
        Input(
            name="phoenix_endpoint",
            value="https://app.phoenix.arize.com",
            env_var="ENDPOINT",
        ),  # TODO: Endpoint (can we add is_endpoint to the config so we can detect and show the link in the UI?)
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


arize_fastapi_app = FastAPIApp(
    name="rag-fastapi-arize",
    lifespan=lifespan,
    inputs=[
        Input(
            name="vector_db",
            value=VectorDB.query(),
            download=True,
            mount="/root/vector_db",
        ),
        Input(
            name="arize-space-id", value="<YOUR_SPACE_ID>", env_var="ARIZE_SPACE_ID"
        ),  # TODO: Arize Space ID
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
)  # TODO: needs=[llm_app] (is this ephemeral?)


@arize_fastapi_app.get("/query_rag")
async def query_rag(prompt: str) -> str:
    from llama_index.core import Settings, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        base_url=f"{deepseek_app.query_endpoint(public=False)}/v1",
        api_key="random",
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EmbeddingConfig.model_name)

    storage_context = StorageContext.from_defaults(persist_dir="/root/vector_db")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()

    response = query_engine.query(prompt)
    return response
