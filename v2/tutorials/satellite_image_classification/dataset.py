"""
Dataset utilities for satellite image classification.

Loads EuroSAT dataset from HuggingFace with proper data augmentation
and sampling strategies.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torchvision import transforms


class EuroSATDataset(torch.utils.data.Dataset):
    """
    Wrapper for EuroSAT dataset from HuggingFace.
    
    EuroSAT: Land Use Classification with Sentinel-2
    - 27,000 labeled Sentinel-2 satellite images
    - 10 classes: Industrial, River, SeaLake, Glacier, Forest, Pasture,
                  Permanent_Crop, Annual_Crop, HerbaceousVegetation, Urban
    - 64x64 pixel RGB images
    """

    # Class names for EuroSAT dataset
    CLASS_NAMES = [
        "Annual_Crop",
        "Forest",
        "Herbaceous_Vegetation",
        "Industrial",
        "Pasture",
        "Permanent_Crop",
        "Residential",
        "River",
        "SeaLake",
        "Glacier",
    ]

    def __init__(
        self,
        split: str = "train",
        image_size: int = 64,
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize EuroSAT dataset.
        
        Args:
            split: Which split to load ('train' or 'validation')
            image_size: Target image size (for resizing)
            transform: Optional torchvision transforms to apply
            cache_dir: Directory to cache downloaded dataset
            max_samples: Limit dataset to N samples (useful for testing)
        """
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.max_samples = max_samples

        # Load dataset from HuggingFace
        dataset = load_dataset(
            "timm/eurosat-rgb",
            cache_dir=cache_dir,
        )[split]

        # Limit to max_samples if specified
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        # Pre-load all images into RAM using batch Arrow reads (much faster than row-by-row).
        # 27k images at 64x64 RGB ≈ 320 MB — negligible vs the 32 GB container limit.
        print(f"Pre-loading {len(dataset)} images into RAM...")
        self.images = []
        self.labels = []
        for i in range(0, len(dataset), 512):
            batch = dataset[i : i + 512]
            for img, label in zip(batch["image"], batch["label"]):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self.images.append(img)
                self.labels.append(label)
        print("Pre-loading complete.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            Tuple of (image tensor, label)
        """
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


def get_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform) for the given image size."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def create_data_loaders(
    image_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for EuroSAT.
    
    Uses proper sampling strategies:
    - RandomSampler for training (shuffles data)
    - SequentialSampler for validation
    
    Args:
        image_size: Size to resize images to
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Fraction of data to use for validation
        cache_dir: Directory to cache downloaded dataset
        max_train_samples: Limit training set
        max_val_samples: Limit validation set
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = get_transforms(image_size)

    # Two dataset objects over the same split, one per transform
    train_dataset = EuroSATDataset(
        split="train",
        image_size=image_size,
        transform=train_transform,
        cache_dir=cache_dir,
        max_samples=max_train_samples,
    )
    val_dataset = EuroSATDataset(
        split="train",
        image_size=image_size,
        transform=val_transform,
        cache_dir=cache_dir,
        max_samples=max_train_samples,
    )

    n_samples = len(train_dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Limit val samples if specified
    if max_val_samples and len(val_subset) > max_val_samples:
        val_indices_limited = np.random.choice(
            len(val_subset), max_val_samples, replace=False
        )
        val_subset = Subset(val_subset, val_indices_limited)

    train_sampler = RandomSampler(train_subset)
    val_sampler = SequentialSampler(val_subset)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    return train_loader, val_loader


def get_class_names() -> list:
    """Get EuroSAT class names."""
    return EuroSATDataset.CLASS_NAMES
