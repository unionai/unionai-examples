import os

import flytekit
import torch
from diffusers import DiffusionPipeline
from flytekit import ImageSpec, Resources, Secret, task
from huggingface_hub import repo_exists

SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
SECRET_KEY = "samhita-hf-token-fjxgnm"

fuse_lora_image = ImageSpec(
    name="fuse_lora_pokemon",
    registry=os.getenv("REGISTRY"),
    packages=[
        "torch==2.2.1",
        "transformers==4.39.1",
        "diffusers==0.27.2",
        "peft==0.10.0",
        "huggingface-hub==0.22.2",
    ],
    python_version="3.12",
)


@task(
    cache=True,
    cache_version="1",
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        )
    ],
    container_image=fuse_lora_image,
    requests=Resources(gpu="1", mem="5Gi"),
)
def fuse_lora(model_name: str, repo_id: str, fused_model_name: str) -> str:
    if not repo_exists(fused_model_name):
        pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
        pipeline.load_lora_weights(repo_id)
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        pipeline.save_pretrained("fused-lora")

        hub_token = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_KEY)
        pipeline.push_to_hub(fused_model_name, token=hub_token)

    return fused_model_name
