"""Task to cache Qwen2.5-Coder model."""

from pathlib import Path
import os
from typing import Annotated
from union import ImageSpec, task, Artifact, FlyteDirectory, current_context, Resources


hf_download_image = ImageSpec(
    name="hfhub-cache",
    packages=["huggingface_hub[hf_transfer]==0.26.3", "union>=0.1.182"],
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)
Qwen_Coder_Artifact = Artifact(name="Qwen2.5-Coder-0.5B")
COMMIT = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"


@task(container_image=hf_download_image, requests=Resources(cpu="2", mem="3Gi", ephemeral_storage="20Gi"))
def cache_model() -> Annotated[FlyteDirectory, Qwen_Coder_Artifact]:
    from huggingface_hub import snapshot_download

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    hf_cache = working_dir / "cache"
    hf_cache.mkdir()

    snapshot_download("Qwen/Qwen2.5-Coder-0.5B", cache_dir=hf_cache, revision=COMMIT)

    snapshot_dir = hf_cache / "models--Qwen--Qwen2.5-Coder-0.5B" / "snapshots" / COMMIT
    assert snapshot_dir.exists()
    return Qwen_Coder_Artifact.create_from(snapshot_dir)
