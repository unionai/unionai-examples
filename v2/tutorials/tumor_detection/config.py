"""
Configuration for brain tumor MRI classification pipeline.

Defines task environments, resource requirements, and training hyperparameters.
"""

import pathlib

import flyte

image = flyte.Image.from_debian_base(
    name="tumor_detection_gpu"
).with_pip_packages(
    "torch",
    "lightning",
    "torchvision",
    "timm",
    "pillow",
    "scikit-learn",
    "plotly",
    "numpy",
    "pandas",
    "torchmetrics",
    "datasets",
).with_source_folder(
    pathlib.Path(__file__).parent,
    copy_contents_only=True,
)

# Downloads raw MRI JPEG files — CPU only, no auth needed, result is cached
dataset_env = flyte.TaskEnvironment(
    name="tumor_dataset",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="8Gi"),
    cache="auto",
)

# GPU training — result is cached so re-running with the same data + config is free
training_env = flyte.TaskEnvironment(
    name="tumor_gpu_training",
    image=image,
    resources=flyte.Resources(
        cpu=8,
        memory="32Gi",
        gpu="T4:1",
        disk="100Gi",
    ),
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_CUDA_MEMORY_FRACTION": "1.0",
    },
    cache="auto",
)

# Report generation — CPU only, reads training results and renders Union UI panels
report_env = flyte.TaskEnvironment(
    name="tumor_report",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Pipeline driver — lightweight orchestrator that calls the three tasks above
pipeline_env = flyte.TaskEnvironment(
    name="tumor_pipeline",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[dataset_env, training_env, report_env],
)


class TrainingConfig:
    """Unified training configuration for brain tumor MRI classification."""

    def __init__(
        self,
        image_size: int = 380,
        num_classes: int = 4,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        phase1_epochs: int = 5,
        phase1_lr: float = 1e-3,
        phase1_freeze_backbone: bool = True,
        phase2_epochs: int = 10,
        phase2_lr: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
        focal_gamma: float = 2.0,
        mixup_alpha: float = 0.2,
        log_interval: int = 50,
    ):
        self.image_size = image_size
        self.num_classes = num_classes

        self.model_name = model_name
        self.pretrained = pretrained

        self.phase1_epochs = phase1_epochs
        self.phase1_lr = phase1_lr
        self.phase1_freeze_backbone = phase1_freeze_backbone

        self.phase2_epochs = phase2_epochs
        self.phase2_lr = phase2_lr

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.focal_gamma = focal_gamma
        self.mixup_alpha = mixup_alpha

        self.log_interval = log_interval

    def to_dict(self) -> dict:
        return self.__dict__
