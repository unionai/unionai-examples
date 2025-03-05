import union
from flytekit.extras.accelerators import A100, L4
from union.app import App, Input

from .frontend.utils import (
    NIMEmbeddingModel,
    NIMLLMModel,
    NIMRerankerModel,
    enterprise_rag_embedding_image,
    enterprise_rag_llm_image,
    enterprise_rag_reranker_image,
)

embedding_model = App(
    name="enterprise-rag-embedding",
    inputs=[
        Input(
            name="enterprise-rag-embedding-model",
            value=NIMEmbeddingModel.query(),
            download=True,
            mount="/opt/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_embedding_image.with_packages(
        "union-runtime==0.1.11"
    ),
    command=["/opt/nvidia/nvidia_entrypoint.sh"],
    requests=union.Resources(cpu="3", mem="16Gi", ephemeral_storage="16Gi", gpu="1"),
    env={"NIM_SERVER_PORT": "8080"},
    accelerator=L4,
)

reranker_model = App(
    name="enterprise-rag-reranker",
    inputs=[
        Input(
            name="enterprise-rag-reranker-model",
            value=NIMRerankerModel.query(),
            download=True,
            mount="/opt/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_reranker_image.with_packages(
        "union-runtime==0.1.11"
    ),
    command=["/opt/nvidia/nvidia_entrypoint.sh"],
    env={"NIM_SERVER_PORT": "8080"},
    requests=union.Resources(cpu="2", mem="16Gi", ephemeral_storage="16Gi"),
)

llm_model = App(
    name="enterprise-rag-llm",
    inputs=[
        Input(
            name="enterprise-rag-llm-model",
            value=NIMLLMModel.query(),
            download=True,
            mount="/opt/nim/.cache",
        )
    ],
    min_replicas=1,
    max_replicas=1,
    port=8080,
    container_image=enterprise_rag_llm_image.with_packages("union-runtime==0.1.11"),
    command=["/opt/nim/start-server.sh"],
    requests=union.Resources(cpu="2", mem="20Gi", ephemeral_storage="16Gi", gpu="1"),
    env={"NIM_SERVER_PORT": "8080"},
    accelerator=A100,
)

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
    ],
    limits=union.Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["./frontend/**"],
    args=["uvicorn", "fastapi_app_server:app", "--port", "8080"],
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
            key="nvidia-api-key",
            env_var="NVIDIA_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
        union.Secret(
            key="union-api-key",
            env_var="UNION_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
)
