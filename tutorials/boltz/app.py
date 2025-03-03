from datetime import timedelta

from union import Resources, ImageSpec
from union.app import App, ScalingMetric, Input
from union import Artifact
from flytekit.extras.accelerators import GPUAccelerator

fastapi_image = ImageSpec(
    name="boltz",
    builder="union",
    packages=[
        "union-runtime==0.1.11",
        "fastapi==0.115.11",
        "pydantic==2.10.6",
        "boltz==0.4.1",
        "uvicorn==0.34.0",
        "python-multipart==0.0.20",
    ],
    apt_packages=["build-essential"],
    # registry="ghcr.io/unionai-oss",
)

boltz_fastapi = App(
    name="boltz-fastapi",
    container_image=fastapi_image,
    limits=Resources(cpu="2", mem="10Gi", gpu="1", ephemeral_storage="50Gi"),
    port=8080,
    include=["./boltz_fastapi.py"],
    args=["uvicorn", "boltz_fastapi:app", "--host", "0.0.0.0", "--port", "8080"],
    env={
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        # "CUDA_VISIBLE_DEVICES": "",
    },
    min_replicas=1,
    max_replicas=3,
    scaledown_after=timedelta(minutes=10),
    scaling_metric=ScalingMetric.RequestRate(1),
    accelerator=GPUAccelerator("nvidia-l40s"),
)

streamlit_image = ImageSpec(
    name="streamlit",
    builder="union",
    packages=[
        "union-runtime==0.1.11",
        "streamlit==1.42.2",
        "pydantic==2.10.6",
        "boltz==0.4.1",
    ],
)

boltz_model = Artifact(
    project="flytesnacks",
    domain="development",
    name="boltz-1",
    version="7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1",
    partitions={
        "task": "auto",
        "model_type": "custom",
        "huggingface-source": "boltz-community/boltz-1",
        "format": "None",
        "architecture": "custom",
        "_u_type": "model",
    },
)

streamlit_app = App(
    name="boltz-streamlit",
    container_image=streamlit_image,
    inputs=[
        Input(
            name="boltz_model", value=boltz_model.query(), download=True, env_var="BOLTZ_MODEL"
        ),
    ],
    args=[
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8080",
        "--server.enableXsrfProtection",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ],
    port=8080,
    limits=Resources(cpu="2", mem="10Gi", ephemeral_storage="50Gi"),
    env={
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "USE_CPU_ONLY": "1",
    },
    include=["./streamlit_app.py"],
)
