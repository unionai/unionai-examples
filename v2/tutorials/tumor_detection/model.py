"""
Model architecture for brain tumor MRI classification.

Uses EfficientNet-B4 (or convnext_small) from timm with pretrained ImageNet weights.
Swap model_name="convnext_small" in TrainingConfig to use ConvNeXt instead.
"""

import math
from typing import Optional, Tuple

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in tumor detection.
    Downweights well-classified examples so training focuses on hard cases.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        gamma_per_class: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        # register_buffer ensures tensors move with .to(device) / fp16 casting
        self.register_buffer("weight", weight)
        self.register_buffer("gamma_per_class", gamma_per_class)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none", label_smoothing=0.1
        )
        pt = torch.exp(-ce_loss)
        if self.gamma_per_class is not None:
            # Use per-class gamma: hard classes (Meningioma) get a higher value
            gamma_t = self.gamma_per_class[targets]
        else:
            gamma_t = self.gamma
        return ((1 - pt) ** gamma_t * ce_loss).mean()


class TumorClassifier(nn.Module):
    """
    EfficientNet-B4 based MRI classifier with feature extraction.

    Supports two-phase training:
    - Phase 1: Freeze backbone, train classifier head only
    - Phase 2: Unfreeze backbone, fine-tune full model with differential LRs
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained backbone from timm, strip its classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class TumorClassifierLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for brain tumor MRI classification.

    Handles focal loss, mixup augmentation, per-class accuracy, and macro F1 logging.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        freeze_backbone: bool = True,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
        max_epochs: int = 15,
        focal_gamma: float = 2.0,
        mixup_alpha: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
        gamma_per_class: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "gamma_per_class"])

        self.model = TumorClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            self.model.freeze_backbone()

        self.criterion = FocalLoss(gamma=focal_gamma, weight=class_weights, gamma_per_class=gamma_per_class)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha

    def on_train_epoch_start(self):
        """Keep frozen backbone in eval mode so BatchNorm uses running stats consistently."""
        if not next(self.model.backbone.parameters()).requires_grad:
            self.model.backbone.eval()

    def _mixup_batch(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup: interpolate pairs of samples and return both label sets + lambda."""
        if self.mixup_alpha <= 0:
            return images, labels, labels, 1.0
        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, labels, labels[index], lam

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        mixed, labels_a, labels_b, lam = self._mixup_batch(images, labels)

        logits = self.model(mixed)
        loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)

        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

        # Stash for per-class metrics computed at epoch end
        if not hasattr(self, "_val_preds"):
            self._val_preds, self._val_targets = [], []
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        """Compute and log per-class accuracy and macro F1 at end of each val epoch."""
        if not hasattr(self, "_val_preds") or not self._val_preds:
            return
        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        self._val_preds.clear()
        self._val_targets.clear()

        for cls_idx in range(self.num_classes):
            mask = targets == cls_idx
            if mask.sum() > 0:
                cls_acc = (preds[mask] == targets[mask]).float().mean()
                self.log(f"val/acc_class_{cls_idx}", cls_acc, sync_dist=True)

        f1_scores = []
        for cls_idx in range(self.num_classes):
            tp = ((preds == cls_idx) & (targets == cls_idx)).sum().float()
            fp = ((preds == cls_idx) & (targets != cls_idx)).sum().float()
            fn = ((preds != cls_idx) & (targets == cls_idx)).sum().float()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_scores.append((2 * precision * recall / (precision + recall + 1e-8)).item())

        self.log("val/macro_f1", sum(f1_scores) / len(f1_scores), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """AdamW with weight-decay separation + linear warmup then cosine decay."""
        decay_params, no_decay_params = [], []
        for param in self.model.parameters():
            if param.requires_grad:
                (decay_params if param.ndim >= 2 else no_decay_params).append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
