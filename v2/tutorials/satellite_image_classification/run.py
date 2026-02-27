"""
Flyte/Union pipeline for satellite image classification.

Multi-file orchestration example demonstrating:
- EfficientNet-B0 transfer learning on EuroSAT
- Two-phase training (frozen backbone + fine-tuning)
- Weights & Biases integration for metrics and t-SNE visualization
- Single GPU training with proper data sampling
- Production-ready pipeline structure
"""


# lightning
# wandb
# datasets
# torchvision
# timm
# plotly
# scikit-learn

import flyte
from flyteplugins.wandb import wandb_config, wandb_init

from config import TrainingConfig, training_env
from training import train_satellite_classifier


# Training configuration
TRAINING_CONFIG = TrainingConfig(
    phase1_epochs=15,
    phase2_epochs=25,
    phase1_lr=2e-3,
    phase2_lr=1e-4,
    batch_size=64,
    num_workers=0,
    log_interval=50,
    tsne_interval=3,
)


@wandb_init
@training_env.task
def satellite_classification_pipeline() -> str:
    """
    Main Flyte pipeline for satellite image classification.

    Orchestrates:
    1. Model training with two-phase learning
    2. Metrics logging to W&B

    Returns:
        Path to best checkpoint
    """
    config = TRAINING_CONFIG

    print("\n" + "=" * 80)
    print("SATELLITE IMAGE CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Phase 1 epochs: {config.phase1_epochs} (frozen backbone)")
    print(f"Phase 2 epochs: {config.phase2_epochs} (fine-tuning)")
    print("=" * 80 + "\n")

    best_checkpoint_path = train_satellite_classifier(
        config=config,
    )

    print(f"\n✓ Pipeline complete!")
    print(f"Best checkpoint: {best_checkpoint_path}")

    return best_checkpoint_path


if __name__ == "__main__":
    # Initialize Flyte configuration
    flyte.init_from_config()

    # Run the pipeline
    print("\nLaunching Satellite Image Classification Pipeline...\n")

    run = flyte.with_runcontext(
        custom_context=wandb_config(
            project=TRAINING_CONFIG.wandb_project,
            entity=TRAINING_CONFIG.wandb_entity,
        ),
    ).run(
        satellite_classification_pipeline,
    )

    print(f"\n✓ Pipeline submitted!")
    print(f"Run URL: {run.url}")
