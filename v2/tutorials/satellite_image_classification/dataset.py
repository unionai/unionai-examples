"""
Dataset utilities for satellite image classification.

Downloads EuroSAT via torchvision and saves raw JPEG files to a flyte.io.Dir
for caching. All preprocessing (resize, normalize, augment) runs per-batch
in the training task via create_data_loaders().
"""

from pathlib import Path
from typing import Tuple

import torch
from flyte.io import Dir
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


async def download_eurosat(
    local_dir: str = "/tmp/eurosat_raw",
) -> Dir:
    """
    Download EuroSAT JPEG files via torchvision and return as flyte.io.Dir.
    No preprocessing — raw images are cached and passed to the training task.
    """
    from torchvision.datasets import EuroSAT as TorchEuroSAT

    print("Downloading EuroSAT dataset...")
    TorchEuroSAT(root=local_dir, download=True)
    print(f"Download complete → {local_dir}")
    return await Dir.from_local(local_dir)


def _find_image_root(dataset_path: Path) -> str:
    """Find the ImageFolder-compatible root inside the downloaded EuroSAT directory."""
    # TorchEuroSAT downloads to root/eurosat/2750/<ClassName>/
    standard = dataset_path / "eurosat" / "2750"
    if standard.exists():
        return str(standard)
    # Fallback: search for directory containing class subdirectories
    matches = list(dataset_path.glob("**/AnnualCrop"))
    if matches:
        return str(matches[0].parent)
    raise RuntimeError(f"Could not find EuroSAT image directory under {dataset_path}")


def create_data_loaders(
    dataset_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load raw EuroSAT JPEG images, apply full preprocessing per-batch,
    split into train/val, and return (train_loader, val_loader).
    """
    from torchvision.datasets import ImageFolder

    image_root = _find_image_root(Path(dataset_path))
    print(f"Loading images from {image_root}")

    train_transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Two ImageFolder instances on the same directory — different transforms per split
    train_full = ImageFolder(image_root, transform=train_transform)
    val_full = ImageFolder(image_root, transform=val_transform)

    n = len(train_full)
    n_val = int(n * val_split)
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=rng).tolist()

    train_dataset = torch.utils.data.Subset(train_full, indices[n_val:])
    val_dataset = torch.utils.data.Subset(val_full, indices[:n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    return train_loader, val_loader


def get_class_names() -> list:
    """Get EuroSAT class names."""
    return CLASS_NAMES
