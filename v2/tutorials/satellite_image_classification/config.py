"""
Configuration for satellite image classification pipeline.

Defines task environments, resource requirements, and training hyperparameters.
"""

import flyte
from dataclasses import dataclass


image = flyte.Image.from_debian_base(
    name="satellite_classification_gpu"
).with_pip_packages(
    "torch",
    "lightning",
    "torchvision",
    "timm",
    "pillow",
    "wandb",
    "flyteplugins-wandb",
    "scikit-learn",
    "plotly",
    "numpy",
    "pandas",
)

# Downloads raw EuroSAT JPEG files — CPU only, result is cached
dataset_env = flyte.TaskEnvironment(
    name="satellite_dataset",
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi", disk="4Gi"),
    cache="auto",
)

# GPU training — result is cached so re-running with the same data + config is free
training_env = flyte.TaskEnvironment(
    name="satellite_gpu_training",
    image=image,
    resources=flyte.Resources(
        cpu=8,
        memory="32Gi",
        gpu="T4:1",
        disk="100Gi",
    ),
    secrets=[flyte.Secret(key="niels_wandb_api_key", as_env_var="WANDB_API_KEY")],
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_CUDA_MEMORY_FRACTION": "1.0",
    },
    cache="auto",
)

# Report generation — CPU only, reads training results and renders Union UI panels
report_env = flyte.TaskEnvironment(
    name="satellite_report",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Pipeline driver — lightweight orchestrator that calls the three tasks above
pipeline_env = flyte.TaskEnvironment(
    name="satellite_pipeline",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[dataset_env, training_env, report_env],
)


@dataclass
class TrainingConfig:
    image_size: int = 224
    num_classes: int = 10
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3
    phase1_freeze_backbone: bool = True
    phase2_epochs: int = 10
    phase2_lr: float = 1e-4
    batch_size: int = 32
    num_workers: int = 0
    val_split: float = 0.1
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    log_interval: int = 100
    tsne_interval: int = 2
    wandb_project: str = "satellite-classification"
    wandb_entity: str = None
