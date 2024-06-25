import flytekit
import safetensors.torch
from flytekit import ImageSpec, Resources, Secret, task
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from .constants import BUILDER, HF_KEY, REGISTRY

lora_image = ImageSpec(
    name="lora_nim",
    registry=REGISTRY,
    packages=["huggingface-hub==0.23.4", "safetensors==0.4.3", "torch==2.3.1"],
    builder=BUILDER,
)


def file_exists(
    repo_id: str,
    filename: str,
) -> bool:
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    try:
        get_hf_file_metadata(url)
        return True
    except (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError):
        return False


@task(
    requests=Resources(mem="7Gi"),
    container_image=lora_image,
    secret_requests=[Secret(key=HF_KEY)],
)
def update_lora(repo_id: str):
    if file_exists(repo_id=repo_id, filename="rest.safetensors"):
        return

    filepath = hf_hub_download(repo_id=repo_id, filename="adapter_model.safetensors")
    tensors = safetensors.torch.load_file(filepath)

    # Separate non-lora keys
    non_lora_keys = [k for k in tensors.keys() if "lora" not in k]

    print("splitting non-lora keys into a separate file")
    print("lora keys: ", tensors.keys())
    print("non-lora keys: ", non_lora_keys)

    non_lora_tensors = {k: tensors.pop(k) for k in non_lora_keys}

    safetensors.torch.save_file(tensors, "adapter_model_updated.safetensors")
    safetensors.torch.save_file(non_lora_tensors, "rest.safetensors")

    api = HfApi(token=flytekit.current_context().secrets.get(HF_KEY))
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj="adapter_model_updated.safetensors",
        path_in_repo="adapter_model.safetensors",
        commit_message="Updated adapter_model with non-lora keys removed",
    )

    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj="rest.safetensors",
        path_in_repo="rest.safetensors",
        commit_message="Added non-lora keys as a separate file",
    )
