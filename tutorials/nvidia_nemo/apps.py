import union
from union.app import App, Input
from flytekit.extras.accelerators import A100

guardrails_image = union.ImageSpec(
    name="nemo-guardrails-server",
    apt_packages=["gcc", "g++"],
    packages=[
        "nemoguardrails==0.13.0",
        "union-runtime>=0.1.11",
    ],
)

llm_image = union.ImageSpec(
    name="enterprise-rag-llm",
    base_image="samhitaalla/enterprise-rag-llm:JTI8JxKjsKSZJ9s8NBLuTQ",
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
    container_image=llm_image,
    args="/opt/nim/start-server.sh",
    requests=union.Resources(cpu="12", mem="60Gi", ephemeral_storage="40Gi", gpu="1"),
    env={
        "NIM_SERVER_PORT": "8080",
        "NIM_CACHE_PATH": "/root/nim/.cache",
        "NGC_HOME": "/root/nim/.cache/ngc/hub",
    },
    accelerator=A100,
)

actions_server = App(
    name="nemo-actions-server",
    container_image=guardrails_image,
    port=8080,
    min_replicas=1,
    max_replicas=1,
    args=[
        "nemoguardrails",
        "actions-server",
        "--port",
        "8080",
    ],
    requests=union.Resources(cpu="2", mem="1Gi"),
)


guardrails_server = App(
    name="nemo-guardrails-server",
    container_image=guardrails_image.with_packages(
        ["langchain-nvidia-ai-endpoints==0.3.9"]
    ),
    inputs=[
        Input(
            name="model_endpoint",
            value=llm_model.query_endpoint(public=False),
            env_var="MODEL_ENDPOINT",
        ),
        Input(
            name="actions_server_url",
            value=actions_server.query_endpoint(public=False),
            env_var="ACTIONS_SERVER_URL",
        ),
    ],
    port=8080,
    min_replicas=1,
    max_replicas=1,
    args=["nemoguardrails", "server", "--config", "config/", "--port", "8080"],
    requests=union.Resources(cpu="2", mem="1Gi"),
    include=["./config/**"],
    secrets=[union.Secret(key="openai_key", env_var="OPENAI_API_KEY")],
)
