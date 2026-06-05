"""
Dataset utilities for brain tumor MRI classification.

Downloads the Brain Tumor MRI Dataset from Hugging Face
(AIOmarRehan/Brain_Tumor_MRI_Dataset — no auth required) and saves images
to a flyte.io.Dir for caching in ImageFolder layout. All preprocessing runs
per-batch in the training task via create_data_loaders().
"""

import shutil
from pathlib import Path
from typing import Tuple

from flyte.io import Dir


# Alphabetical order matches torchvision ImageFolder's class indexing
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Normalize whatever label strings the HF dataset uses to canonical class names
_LABEL_NORM = {
    "glioma": "Glioma", "glioma_tumor": "Glioma",
    "meningioma": "Meningioma", "meningioma_tumor": "Meningioma",
    "no_tumor": "No Tumor", "notumor": "No Tumor", "no tumor": "No Tumor",
    "pituitary": "Pituitary", "pituitary_tumor": "Pituitary",
}


async def download_tumor_dataset(
    local_dir: str = "/tmp/tumor_raw",
) -> Dir:
    """
    Download Brain Tumor MRI dataset from Hugging Face and return as flyte.io.Dir.
    Images are saved in ImageFolder layout: <local_dir>/<ClassName>/<idx>.jpg.
    No auth required — the HF dataset is public.
    """
    from datasets import concatenate_datasets, load_dataset

    print("Downloading AIOmarRehan/Brain_Tumor_MRI_Dataset from Hugging Face...")
    ds = load_dataset("AIOmarRehan/Brain_Tumor_MRI_Dataset")

    # Combine ALL splits — the full dataset has ~7k images across train+test.
    # Splitting into train/val is handled by create_data_loaders, not here.
    all_splits = list(ds.values())
    combined = concatenate_datasets(all_splits) if len(all_splits) > 1 else all_splits[0]
    print(f"Total images across {len(all_splits)} split(s): {len(combined)}")
    if len(combined) < 2000:
        print("WARNING: fewer than 2000 images loaded — the HF dataset may only ")
        print("contain the test split. Consider using the full Kaggle dataset for better results.")

    dest = Path(local_dir)
    if dest.exists():
        shutil.rmtree(dest)

    # Resolve label: HF may give ClassLabel (int) or raw string
    features = combined.features
    label_feature = features.get("label")

    for idx, example in enumerate(combined):
        raw_label = example["label"]
        if hasattr(label_feature, "int2str"):
            label_str = label_feature.int2str(raw_label)
        else:
            label_str = str(raw_label)

        # Normalize to canonical class name (e.g. "glioma_tumor" → "Glioma")
        class_name = _LABEL_NORM.get(label_str.lower().replace(" ", "_"), label_str)

        class_dir = dest / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        img = example["image"]  # PIL Image
        # Convert to RGB — some MRI scans are saved as grayscale or RGBA
        img = img.convert("RGB")
        img.save(str(class_dir / f"{idx:06d}.jpg"), "JPEG", quality=95)

    counts = {cls: len(list((dest / cls).glob("*.jpg"))) for cls in CLASS_NAMES if (dest / cls).exists()}
    print(f"Class distribution: {counts}")
    return await Dir.from_local(local_dir)


def compute_class_weights(dataset_path: str):
    """
    Compute inverse-frequency class weights from the saved ImageFolder layout.
    Returns a float32 tensor of shape (num_classes,) for use in focal loss.
    """
    import numpy as np
    import torch
    from torchvision.datasets import ImageFolder

    image_root = _find_image_root(Path(dataset_path))
    dataset = ImageFolder(image_root)
    counts = np.bincount(
        [label for _, label in dataset.samples], minlength=len(CLASS_NAMES)
    )
    print(f"Class counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    weights = 1.0 / (counts + 1e-6)
    # Meningioma is visually similar to Glioma and consistently the hardest class —
    # boost its weight by 4x beyond inverse-frequency to force the loss to prioritize it.
    meningioma_idx = CLASS_NAMES.index("Meningioma")
    weights[meningioma_idx] *= 4.0
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.tensor(weights, dtype=torch.float32)


def _find_image_root(dataset_path: Path) -> str:
    """Find the ImageFolder-compatible root — the directory that contains the class subdirs."""
    # After download_tumor_dataset, class dirs sit directly under dataset_path
    for cls in CLASS_NAMES:
        if (dataset_path / cls).exists():
            return str(dataset_path)
    # Fallback: search one level deeper
    for child in dataset_path.iterdir():
        if child.is_dir() and any((child / cls).exists() for cls in CLASS_NAMES):
            return str(child)
    raise RuntimeError(f"Could not find class directories under {dataset_path}")


def create_data_loaders(
    dataset_path: str,
    image_size: int = 380,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
) -> Tuple:
    """
    Load raw MRI JPEG images, apply full preprocessing per-batch,
    split into train/val, and return (train_loader, val_loader).
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    image_root = _find_image_root(Path(dataset_path))
    print(f"Loading images from {image_root}")

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
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

    train_indices = indices[n_val:]
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_full, indices[:n_val])

    # WeightedRandomSampler ensures every batch sees balanced classes.
    # Critical for Meningioma/Pituitary which have fewer training samples.
    train_labels = [train_full.targets[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    class_sample_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([class_sample_weights[lbl] for lbl in train_labels])
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"Train class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
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
    return CLASS_NAMES
