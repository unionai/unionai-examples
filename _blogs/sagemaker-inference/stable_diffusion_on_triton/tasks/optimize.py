import os
import shutil
import subprocess
import tarfile
from typing import Annotated

import flytekit
from flytekit import ImageSpec, Resources, task
from flytekit.core.artifact import Inputs
from flytekit.extras.accelerators import A10G
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from .utils import ModelArtifact

sd_compilation_image = ImageSpec(
    name="sd_optimization",
    registry=os.getenv("REGISTRY"),
    packages=[
        "torch==2.2.1",
        "transformers==4.39.1",
        "transformers[onnxruntime]==4.39.1",
        "diffusers==0.27.2",
        "ftfy==6.2.0",
        "scipy==1.12.0",
        "flytekit>=1.13.0",
        "accelerate==0.28.0",
        "peft==0.10.0",
        "union==0.1.46",
    ],
    python_version="3.12",
    source_root="stable_diffusion_on_triton/backend",
    base_image="nvcr.io/nvidia/tensorrt:23.12-py3",
    builder="fast-builder",  # not using "default" builder as it doesn't consider python packages in the base image.
).with_commands(
    [
        "/usr/src/tensorrt/bin/trtexec --help",  # check if trtexec is available
        "chmod +x /root/export.sh",
    ]
)


def generate_md_contents(dataset: str) -> str:
    contents = "# Tensorrt text2image \n" "\n"
    contents += (
        f"This is an optimized tensorrt model that has been trained on {dataset} dataset."
        "\n\n"
        "GPU: A10G"
    )
    return contents


@task(
    cache=True,
    cache_version="2.9",
    container_image=sd_compilation_image,
    requests=Resources(gpu="1", mem="20Gi"),
    accelerator=A10G,
)
def optimize_model(
    fused_lora: FlyteDirectory,  # fused_lora: FlyteDirectory = ModelArtifact.query(dataset=Inputs.dataset, type="fused-lora")
    dataset: str,
) -> Annotated[FlyteDirectory, ModelArtifact(dataset=Inputs.dataset, type="tensorrt")]:
    from union.artifacts import ModelCard

    model_repository = flytekit.current_context().working_directory
    vae_dir = os.path.join(model_repository, "vae")
    encoder_dir = os.path.join(model_repository, "text_encoder")
    pipeline_dir = os.path.join(model_repository, "pipeline")

    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(pipeline_dir, exist_ok=True)

    vae_1_dir = os.path.join(vae_dir, "1")
    encoder_1_dir = os.path.join(encoder_dir, "1")

    os.makedirs(vae_1_dir, exist_ok=True)
    os.makedirs(encoder_1_dir, exist_ok=True)

    vae_plan = os.path.join(vae_1_dir, "model.plan")
    encoder_onnx = os.path.join(encoder_1_dir, "model.onnx")

    fused_lora_path = fused_lora.download()

    result = subprocess.run(
        f"/root/export.sh {vae_plan} {encoder_onnx} {fused_lora_path}",
        capture_output=True,
        text=True,
        shell=True,
    )

    # Check the return code
    if result.returncode == 0:
        print("Script execution succeeded")
        print(f"stdout: {result.stdout}")
    else:
        print("Script execution failed")
        print(f"stderr: {result.stderr}")

    shutil.copy("/root/vae_config.pbtxt", os.path.join(vae_dir, "config.pbtxt"))
    shutil.copy(
        "/root/text_encoder_config.pbtxt",
        os.path.join(encoder_dir, "config.pbtxt"),
    )
    shutil.copytree(
        fused_lora_path, os.path.join(pipeline_dir, "fused-lora"), dirs_exist_ok=True
    )
    shutil.copytree("/root/pipeline", pipeline_dir, dirs_exist_ok=True)

    return ModelArtifact.create_from(
        FlyteDirectory(model_repository),
        ModelCard(generate_md_contents(dataset=dataset)),
    )


@task(cache=True, cache_version="2", requests=Resources(mem="5Gi"))
def compress_model(
    model_repo: FlyteDirectory,  # model_repo: FlyteDirectory = ModelArtifact.query(dataset=Inputs.dataset, type="tensorrt")
    dataset: str,
) -> Annotated[
    FlyteFile, ModelArtifact(dataset=Inputs.dataset, type="sagemaker-compressed-model")
]:
    model_file_name = "stable-diff-bls.tar.gz"

    with tarfile.open(model_file_name, mode="w:gz") as tar:
        tar.add(model_repo.download(), arcname=".")

    return FlyteFile(model_file_name)
