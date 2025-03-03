import os
import subprocess
from time import sleep
from union import (
    workflow,
    FlyteFile,
    ImageSpec,
    Resources,
    FlyteDirectory,
    ActorEnvironment,
)

image = ImageSpec(
    name="boltz",
    # builder="union",
    registry="docker.io/unionbio",
    packages=[
        "union",
        "flytekit==1.13.14",
        "union-runtime==0.1.11",
        "fastapi==0.115.11",
        "pydantic==2.10.6",
        "boltz==0.4.1",
        "uvicorn==0.34.0",
        "python-multipart==0.0.20",
    ],
    apt_packages=["build-essential"],
)

actor = ActorEnvironment(
    name="boltz-actor",
    replica_count=1,
    ttl_seconds=600,
    requests=Resources(
        cpu="2",
        mem="10Gi",
        gpu="1",
    ),
    container_image=image,
)


@actor.task
def check_pytorch_gpu() -> bool:
    import torch

    return torch.cuda.is_available()


@actor.task
def simple_predict(input: FlyteFile) -> FlyteDirectory:
    input.download()
    out = "/tmp/boltz_out"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["boltz", "predict", input.path, "--out_dir", out, "--use_msa_server"])
    return FlyteDirectory(path=out)


@workflow
def wf(input: FlyteFile) -> FlyteDirectory:
    check_pytorch_gpu()
    return simple_predict(input=input)
