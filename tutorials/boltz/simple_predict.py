# Imports and init remote
import os
import subprocess
from union import task, workflow, FlyteFile, ImageSpec, Resources, FlyteDirectory

# Define Image
image = ImageSpec(
    name="boltz",
    builder="union",
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


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="10Gi", ephemeral_storage="50Gi", gpu="1"),
)
def simple_predict(input: FlyteFile) -> FlyteDirectory:
    input.download()
    out = "/tmp/boltz_out"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["boltz", "predict", input.path, "--out_dir", out, "--use_msa_server"])
    return FlyteDirectory(path=out)


@workflow
def wf(input: FlyteFile) -> FlyteDirectory:
    return simple_predict(input=input)
