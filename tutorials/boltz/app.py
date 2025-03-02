from datetime import timedelta

from union import Resources, ImageSpec
from union.app import App, ScalingMetric

image = ImageSpec(
    name="boltz",
    builder="union",
    packages=[
        "union-runtime==0.1.11",
        "fastapi==0.115.11", 
        "pydantic==2.10.6", 
        "boltz==0.4.1", 
        "uvicorn==0.34.0", 
        "python-multipart==0.0.20"
        ],
    apt_packages=["build-essential"]
    # registry="ghcr.io/unionai-oss",
)

boltz_fastapi = App(
    name="boltz-fastapi",
    container_image=image,
    limits=Resources(cpu="2", mem="16Gi", ephemeral_storage="20Gi"),
    port=8080,
    include=["./boltz_fastapi.py"],
    args=["uvicorn", "boltz_fastapi:app", "--host", "0.0.0.0", "--port", "8080"],
    env={
        "PYTORCH_ENABLE_MPS_FALLBACK": "1"
    },
    min_replicas=1,
    max_replicas=3,
    scaledown_after=timedelta(minutes=10),
    scaling_metric=ScalingMetric.RequestRate(1),
)
