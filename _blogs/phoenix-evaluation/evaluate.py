import flytekit as fl
import union
from flytekit import Resources
from flytekit.extras.accelerators import A10G
from union.app import App, Input

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
    limits=Resources(cpu="2", mem="20Gi", gpu="1", ephemeral_storage="10Gi"),
    requests=Resources(cpu="2", mem="20Gi", gpu="1", ephemeral_storage="10Gi"),
    env={
        "DEBUG": "1",
        "LOG_LEVEL": "DEBUG",
    },
    accelerator=A10G,
)

gradio_app = App(
    name="vllm-deepseek-gradio",
    inputs=[
        Input(
            name="vllm_deepseek_endpoint",
            value=deepseek_app.query_endpoint(public=False),
            env_var="VLLM_DEEPSEEK_ENDPOINT",
        )
    ],
    container_image=union.ImageSpec(
        name="vllm-deepseek-gradio",
        registry="samhitaalla",
        packages=[
            "gradio",
            "union-runtime>=0.1.10",
            "openinference-instrumentation-openai",
            "openai",
            "git+https://github.com/flyteorg/flytekit.git@4208a641debb0334c49c9331bcc4d98ed5c45d12",
        ],
        apt_packages=["git"],
    ),
    limits=union.Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["gradio_app.py"],
    args=[
        "python",
        "gradio_app.py",
    ],
    min_replicas=1,
    max_replicas=1,
    secrets=[
        fl.Secret(
            key="samhita-phoenix-api-key",
            env_var="PHOENIX_API_KEY",
            mount_requirement=union.Secret.MountType.ENV_VAR,
        ),
    ],
)
