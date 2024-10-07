from flytekit import ImageSpec
from flytekit.core.artifact import Artifact

ModelArtifact = Artifact(
    name="finetuned-model-for-mlc-deployment", partition_keys=["model", "dataset"]
)

download_artifacts_image = ImageSpec(
    name="download-dataset-and-model",
    packages=[
        "datasets==3.0.0",
        "huggingface-hub==0.25.1",
        "flytekitplugins-wandb==1.13.5",
    ],
)

model_training_image = ImageSpec(
    name="llama-3-finetuning",
    packages=[
        "flytekit==1.13.5",
        "datasets==3.0.0",
        "flytekitplugins-wandb==1.13.5",
        "transformers==4.44.2",
        "peft==0.12.0",
        "bitsandbytes==0.44.0",
        "accelerate==0.34.2",
        "trl==0.11.1",
        "torch==2.4.1",
    ],
    cuda="12.1",
)

llm_mlc_image = ImageSpec(
    name="llama-3-8b-llm-mlc",
    packages=[
        "--pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu121 mlc-ai-nightly-cu121",
        "flytekitplugins-wandb==1.13.5",
    ],
    apt_packages=["cmake", "git-lfs", "cargo", "rustc"],
    conda_channels=["nvidia"],
    conda_packages=[
        "cuda=12.1.0",
        "cuda-nvcc",
        "cuda-version=12.1.0",
        "cuda-command-line-tools=12.1.0",
    ],
    python_version="3.12",
)

modelcard_image = ImageSpec(
    name="llama-3-8b-llm-mlc-modelcard",
    packages=["huggingface-hub==0.25.1", "flytekitplugins-wandb==1.13.5"],
    apt_packages=["git", "git-lfs"],
)
