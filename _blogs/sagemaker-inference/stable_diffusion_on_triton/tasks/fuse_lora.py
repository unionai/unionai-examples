import os
from typing import Annotated

import torch
from diffusers import DiffusionPipeline
from flytekit import ImageSpec, Resources, task
from flytekit.core.artifact import Inputs
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from union.artifacts import ModelCard

from .utils import ModelArtifact

fuse_lora_image = ImageSpec(
    name="fuse_lora_pokemon",
    registry=os.getenv("REGISTRY"),
    packages=[
        "torch==2.2.1",
        "transformers==4.39.1",
        "diffusers==0.27.2",
        "peft==0.10.0",
        "union==0.1.46",
        "numpy<2.0.0",
    ],
    python_version="3.12",
    builder="fast-builder",
)


def generate_md_contents(repo_id: str, dataset: str) -> str:
    contents = "# Fused LoRA text2image \n" "\n"
    contents += (
        f"These are fused LoRA adaption weights for {repo_id}. The weights were fine-tuned on the {dataset} dataset."
        "\n\n"
        "GPU: T4"
    )
    return contents


@task(
    cache=True,
    cache_version="1.1",
    container_image=fuse_lora_image,
    requests=Resources(gpu="1", mem="5Gi"),
    accelerator=T4,
)
def fuse_lora(
    repo_id: str,
    lora: FlyteDirectory,  # lora: FlyteDirectory = ModelArtifact.query(dataset=Inputs.dataset, type="lora")
    dataset: str,
) -> Annotated[
    FlyteDirectory, ModelArtifact(dataset=Inputs.dataset, type="fused-lora")
]:
    pipeline = DiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipeline.load_lora_weights(lora.download())
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    pipeline.save_pretrained("fused-lora")

    return ModelArtifact.create_from(
        FlyteDirectory("fused-lora"),
        ModelCard(generate_md_contents(repo_id=repo_id, dataset=dataset)),
    )
