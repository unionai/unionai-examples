"""
Model architecture for satellite image classification.

Uses EfficientNet-B0 from timm with pretrained ImageNet weights.
"""

import math
from typing import Tuple

import lightning as L
import timm
import torch
import torch.nn as nn


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based image classifier with feature extraction.
    
    Supports two-phase training:
    - Phase 1: Freeze backbone, train classifier head
    - Phase 2: Unfreeze backbone, fine-tune full model
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained EfficientNet-B0 from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, we'll add our own
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns logits."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (useful for t-SNE visualization).
        
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        features = self.backbone(x)
        return features

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_frozen_params(self) -> list:
        """Get list of frozen parameters."""
        return [p for p in self.backbone.parameters() if not p.requires_grad]

    def get_trainable_params(self) -> list:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


class SatelliteClassifierLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for satellite image classification.
    
    Handles:
    - Loss computation
    - Metrics logging
    - Feature extraction for t-SNE
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        freeze_backbone: bool = True,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
        )

        # Initialize training state
        if freeze_backbone:
            self.model.freeze_backbone()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs

        # For collecting validation features for t-SNE
        self.val_features = []
        self.val_pred_labels = []

    def on_train_epoch_start(self):
        """Keep frozen backbone in eval mode so BatchNorm uses running stats consistently."""
        backbone_frozen = not next(self.model.backbone.parameters()).requires_grad
        if backbone_frozen:
            self.model.backbone.eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(batch)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Log metrics
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Log metrics
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

        # Collect validation features + predicted labels for t-SNE
        features = self.model.extract_features(images).detach().cpu()
        self.val_features.append(features)
        self.val_pred_labels.append(logits.argmax(dim=1).detach().cpu())

        return loss

    def configure_optimizers(self):
        """Configure optimizer with optional warmup schedule."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        for param in self.model.parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Warmup then cosine decay to 0 over the full training run
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = (current_step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_val_features_for_tsne(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get collected validation features and predicted labels for t-SNE."""
        if self.val_features:
            features = torch.cat(self.val_features, dim=0)
            pred_labels = torch.cat(self.val_pred_labels, dim=0)
            self.val_features.clear()
            self.val_pred_labels.clear()
            return features, pred_labels
        return torch.tensor([]), torch.tensor([])
