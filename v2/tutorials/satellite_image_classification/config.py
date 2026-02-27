"""
Configuration for satellite image classification pipeline.

Defines task environments, resource requirements, and training hyperparameters.
"""

import flyte

# ============================================================================
# Task Environment Setup and Container Image
# ============================================================================

# Container image with all dependencies
image = flyte.Image.from_debian_base(
    name="satellite_classification_gpu"
).with_pip_packages(
    # Core ML frameworks
    "torch",
    "lightning",
    "torchvision",
    # Model architecture and pretrained weights
    "timm",
    # Dataset and data loading
    "datasets",
    "pillow",
    # Visualization and reporting
    "wandb",
    "flyteplugins-wandb",
    "scikit-learn",
    "plotly",
    # Utilities
    "numpy",
)

# GPU training environment - single GPU
training_env = flyte.TaskEnvironment(
    name="satellite_gpu_training",
    image=image,
    resources=flyte.Resources(
        cpu=8,
        memory="32Gi",
        gpu="T4:1",
        disk="100Gi",
    ),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_CUDA_MEMORY_FRACTION": "1.0",
    },
    cache="disable",
)


# ============================================================================
# Training Hyperparameters and Configurations
# ============================================================================

class TrainingConfig:
    """Unified training configuration."""

    def __init__(
        self,
        # Dataset
        dataset_name: str = "timm/eurosat-rgb",
        image_size: int = 224,
        num_classes: int = 10,
        # Model
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        # Phase 1: Feature extraction (frozen backbone)
        phase1_epochs: int = 5,
        phase1_lr: float = 1e-3,
        phase1_freeze_backbone: bool = True,
        # Phase 2: Fine-tuning (unfrozen backbone)
        phase2_epochs: int = 10,
        phase2_lr: float = 1e-4,
        phase2_freeze_backbone: bool = False,
        # Data
        batch_size: int = 32,
        num_workers: int = 2,
        val_split: float = 0.1,
        # Optimization
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        # Logging
        log_interval: int = 100,
        tsne_interval: int = 2,  # Update t-SNE every N epochs
        # W&B
        wandb_project: str = "satellite-classification",
        wandb_entity: str = None,
        # Checkpoint
        save_interval: int = 5,  # Save every N epochs
    ):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.phase1_epochs = phase1_epochs
        self.phase1_lr = phase1_lr
        self.phase1_freeze_backbone = phase1_freeze_backbone
        
        self.phase2_epochs = phase2_epochs
        self.phase2_lr = phase2_lr
        self.phase2_freeze_backbone = phase2_freeze_backbone
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        
        self.log_interval = log_interval
        self.tsne_interval = tsne_interval
        
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        self.save_interval = save_interval

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return self.__dict__
