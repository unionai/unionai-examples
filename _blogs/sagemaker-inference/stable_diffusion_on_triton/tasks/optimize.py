import os
import shutil
import subprocess
import tarfile

import flytekit
from flytekit import ImageSpec, Resources, task, Secret
from flytekit.extras.accelerators import A10G
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
SECRET_KEY = "samhita-hf-token-fjxgnm"

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
        "flytekit==1.11.0",
        "accelerate==0.28.0",
        "peft==0.10.0",
    ],
    python_version="3.12",
    source_root="stable_diffusion_on_triton/backend",
    base_image="nvcr.io/nvidia/tensorrt:23.12-py3",
).with_commands(
    [
        "/usr/src/tensorrt/bin/trtexec --help",  # check if trtexec is available
        "chmod +x /root/export.sh",
    ]
)


@task(
    cache=True,
    cache_version="2.6",
    container_image=sd_compilation_image,
    requests=Resources(gpu="1", mem="20Gi"),
    accelerator=A10G,
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        )
    ],
)
def optimize_model(model_name: str, repo_id: str) -> FlyteDirectory:
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

    hub_token = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_KEY)

    result = subprocess.run(
        f"/root/export.sh {vae_plan} {encoder_onnx} {repo_id} {model_name} {hub_token}",
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
    shutil.copytree("/root/pipeline", pipeline_dir, dirs_exist_ok=True)

    return FlyteDirectory(model_repository)


@task(cache=True, cache_version="2", requests=Resources(mem="5Gi"))
def compress_model(model_repo: FlyteDirectory) -> FlyteFile:
    model_file_name = "stable-diff-bls.tar.gz"

    with tarfile.open(model_file_name, mode="w:gz") as tar:
        tar.add(model_repo.download(), arcname=".")

    return FlyteFile(model_file_name)
