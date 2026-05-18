"""
Training pipeline for brain tumor MRI classification.

Implements two-phase training:
- Phase 1: Frozen backbone (feature extractor), train classification head
- Phase 2: Fine-tune full model with differential LRs + cosine annealing
"""

from config import TrainingConfig
from dataset import compute_class_weights, create_data_loaders
from model import TumorClassifierLightningModule
from utils import get_model_size, get_trainable_params


def train_tumor_classifier(
    config: TrainingConfig,
    dataset_path: str,
) -> dict:
    """
    Run two-phase training on the preprocessed dataset and return metrics + final predictions.

    dataset_path: local directory where the flyte.io.Dir was downloaded by the training task.
    """
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    class MetricsLoggerCallback(L.Callback):
        def __init__(self, phase1_epochs: int):
            super().__init__()
            self.phase1_epochs = phase1_epochs
            self.history = []

        def on_validation_epoch_end(self, trainer, _pl_module):
            epoch = trainer.current_epoch
            metrics = trainer.callback_metrics
            self.history.append({
                "epoch": epoch,
                "phase": 1 if epoch < self.phase1_epochs else 2,
                "train_loss": float(metrics.get("train/loss_epoch", 0)),
                "val_loss": float(metrics.get("val/loss", 0)),
                "val_acc": float(metrics.get("val/acc", 0)),
                "macro_f1": float(metrics.get("val/macro_f1", 0)),
            })

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

                print(f"Phase 2 started: lr={self.phase2_lr}")
                print(f"Total parameters: {get_model_size(pl_module.model):,}")
                print(f"Trainable parameters: {get_trainable_params(pl_module.model):,}")
                self.phase_changed = True


    print("\n" + "=" * 80)
    print("BRAIN TUMOR MRI CLASSIFICATION WITH EFFICIENTNET-B4")
    print("=" * 80)
    print(f"Config: {config.to_dict()}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\nLoading MRI images...")
    train_loader, val_loader = create_data_loaders(
        dataset_path=dataset_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
    )
    print(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

    print("\nComputing class weights for focal loss...")
    class_weights = compute_class_weights(dataset_path)
    print(f"Class weights: {class_weights.tolist()}")

    # Per-class gamma: Meningioma gets 7.0, all others 3.0.
    # CLASS_NAMES alphabetical order: Glioma=0, Meningioma=1, No Tumor=2, Pituitary=3
    gamma_per_class = torch.tensor([3.0, 7.0, 3.0, 3.0])

    print("\nInitializing model...")
    model = TumorClassifierLightningModule(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained,
        learning_rate=config.phase1_lr,
        freeze_backbone=config.phase1_freeze_backbone,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_epochs=config.phase1_epochs + config.phase2_epochs,
        focal_gamma=config.focal_gamma,
        mixup_alpha=config.mixup_alpha,
        class_weights=class_weights,
        gamma_per_class=gamma_per_class,
    )

    print(f"Model: {config.model_name}")
    print(f"Total parameters: {get_model_size(model.model):,}")
    print(f"Trainable parameters: {get_trainable_params(model.model):,}")

    from pathlib import Path
    checkpoint_dir = Path("/tmp/tumor_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_cb = MetricsLoggerCallback(phase1_epochs=config.phase1_epochs)

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
        metrics_cb,
        PhaseChangeCallback(
            phase1_epochs=config.phase1_epochs,
            phase2_lr=config.phase2_lr,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=config.phase1_epochs + config.phase2_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=config.log_interval,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"\n✓ Training complete!")
    print(f"Best checkpoint: {best_checkpoint}")

    # Final inference with TTA (test-time augmentation): average logits over
    # original + h-flip + v-flip + 90° rotations for a free accuracy boost.
    print("\nRunning final inference with TTA for confusion matrix...")
    import numpy as np
    import torchvision.transforms.functional as TF
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            aug_logits = [
                model.model(images),
                model.model(TF.hflip(images)),
                model.model(TF.vflip(images)),
                model.model(torch.rot90(images, k=1, dims=[2, 3])),
                model.model(torch.rot90(images, k=3, dims=[2, 3])),
            ]
            avg_logits = torch.stack(aug_logits).mean(dim=0)
            all_preds.append(avg_logits.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
    final_preds = torch.cat(all_preds).numpy()
    final_targets = torch.cat(all_targets).numpy()

    return {
        "best_checkpoint": best_checkpoint,
        "metrics": metrics_cb.history,
        "final_preds": final_preds.tolist(),
        "final_targets": final_targets.tolist(),
    }
