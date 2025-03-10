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
