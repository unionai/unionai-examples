"""
Training pipeline for satellite image classification.

Implements two-phase training:
- Phase 1: Frozen backbone (feature extractor), train classification head
- Phase 2: Fine-tune backbone with lower learning rate
"""

from pathlib import Path
from typing import Optional

from flyteplugins.wandb import get_wandb_run

from config import TrainingConfig
from dataset import create_data_loaders, get_class_names
from model import SatelliteClassifierLightningModule
from utils import get_model_size, get_trainable_params, log_tsne_to_wandb


def train_satellite_classifier(
    config: TrainingConfig,
) -> str:
    """Run two-phase training and return the best checkpoint path."""
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    class TsneCallback(L.Callback):
        def __init__(
            self,
            interval: int = 2,
            n_components: int = 2,
            class_names: Optional[list] = None,
        ):
            super().__init__()
            self.interval = interval
            self.n_components = n_components
            self.class_names = class_names or []

        def on_validation_epoch_end(self, trainer, pl_module):
            # Always drain features so they never accumulate across epochs
            val_features, val_pred_labels = pl_module.get_val_features_for_tsne()

            epoch = trainer.current_epoch
            should_log = (epoch == 0) or ((epoch + 1) % self.interval == 0)

            if should_log and len(val_features) > 0:
                log_tsne_to_wandb(
                    val_features,
                    val_pred_labels,
                    self.class_names,
                    split="predicted",
                    epoch=epoch,
                    n_components=self.n_components,
                )
                print(f"t-SNE visualization logged for epoch {epoch}")

    class PhaseChangeCallback(L.Callback):
        def __init__(self, phase1_epochs: int, phase2_lr: float):
            super().__init__()
            self.phase1_epochs = phase1_epochs
            self.phase2_lr = phase2_lr
            self.phase_changed = False

        def on_train_epoch_end(self, trainer, pl_module):
            if not self.phase_changed and (trainer.current_epoch + 1) == self.phase1_epochs:
                print("\n" + "=" * 80)
                print("TRANSITIONING TO PHASE 2: UNFREEZING BACKBONE AND ADJUSTING LR")
                print("=" * 80 + "\n")

                pl_module.model.unfreeze_backbone()

                # Set classifier param groups to phase2_lr
                for param_group in trainer.optimizers[0].param_groups:
                    param_group["lr"] = self.phase2_lr

                # Add backbone params to optimizer with 10x lower LR.
                # Backbone was excluded at init because it was frozen.
                backbone_lr = self.phase2_lr * 0.1
                backbone_decay, backbone_no_decay = [], []
                for param in pl_module.model.backbone.parameters():
                    if param.ndim >= 2:
                        backbone_decay.append(param)
                    else:
                        backbone_no_decay.append(param)
                optimizer = trainer.optimizers[0]
                optimizer.add_param_group({"params": backbone_decay, "lr": backbone_lr, "weight_decay": pl_module.weight_decay})
                optimizer.add_param_group({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0})

                # Fresh cosine schedule over remaining Phase 2 steps to avoid
                # the Phase 1 schedule arriving near-zero before Phase 2 begins.
                steps_remaining = trainer.estimated_stepping_batches - trainer.global_step
                new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizers[0],
                    T_max=max(1, steps_remaining),
                    eta_min=1e-6,
                )
                for lr_scheduler_config in trainer.lr_scheduler_configs:
                    lr_scheduler_config.scheduler = new_scheduler

                get_wandb_run().log({"phase": 2, "learning_rate": self.phase2_lr, "epoch": trainer.current_epoch})
                print(f"Total parameters: {get_model_size(pl_module.model):,}")
                print(f"Trainable parameters: {get_trainable_params(pl_module.model):,}")
                self.phase_changed = True

    print("\n" + "=" * 80)
    print("SATELLITE IMAGE CLASSIFICATION WITH EFFICIENTNET-B0")
    print("=" * 80)
    print(f"Config: {config.to_dict()}\n")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\nLoading EuroSAT dataset...")
    train_loader, val_loader = create_data_loaders(
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        cache_dir="/tmp/eurosat_cache",
    )
    print(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

    class_names = get_class_names()

    print("\nInitializing model...")
    model = SatelliteClassifierLightningModule(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained,
        learning_rate=config.phase1_lr,
        freeze_backbone=config.phase1_freeze_backbone,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_epochs=config.phase1_epochs + config.phase2_epochs,
    )

    print(f"Model: {config.model_name}")
    print(f"Total parameters: {get_model_size(model.model):,}")
    print(f"Trainable parameters: {get_trainable_params(model.model):,}")

    checkpoint_dir = Path("/tmp/satellite_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:03d}-{val_acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            verbose=True,
            auto_insert_metric_name=False,
        ),
        TsneCallback(
            interval=config.tsne_interval,
            n_components=2,
            class_names=class_names,
        ),
        PhaseChangeCallback(
            phase1_epochs=config.phase1_epochs,
            phase2_lr=config.phase2_lr,
        ),
    ]

    wandb_logger = WandbLogger(experiment=get_wandb_run(), log_model=False)

    print("\n" + "=" * 80)
    print(f"Phase 1 ({config.phase1_epochs} epochs): frozen backbone, lr={config.phase1_lr}")
    print(f"Phase 2 ({config.phase2_epochs} epochs): fine-tune backbone, lr={config.phase2_lr}")
    print("=" * 80 + "\n")

    trainer = L.Trainer(
        max_epochs=config.phase1_epochs + config.phase2_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=config.log_interval,
    )

    trainer.fit(model, train_loader, val_loader)

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"\n✓ Training complete!")
    print(f"Best checkpoint: {best_checkpoint}")

    print("\nRunning final validation...")
    trainer.validate(model, val_loader)

    get_wandb_run().log(
        {
            "best_val_acc": trainer.checkpoint_callback.best_model_score.item(),
            "training_complete": True,
        }
    )

    return best_checkpoint
