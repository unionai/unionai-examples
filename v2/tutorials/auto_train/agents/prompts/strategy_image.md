**Image strategy** (N={img_n:,} samples, native resolution and channels detected at runtime by the data skeleton):

Variables already in scope from the skeleton: `dataset` (ImageFolder), `train_idx`, `val_idx`, `native_h`, `native_w`, `img_channels`, `num_classes`, `train_transform`, `val_transform`. Build DataLoaders with `SubsetRandomSampler(train_idx)` / `SubsetRandomSampler(val_idx)` — do NOT re-split.

Scale-based strategy:
- **N < 1,000** — freeze backbone completely. Extract embeddings once (FP16, no_grad, batch=256), cache to disk. Train a sklearn LogisticRegression or SVC on the cached embeddings.
- **1,000 ≤ N < 10,000** — lightweight ImageNet-pretrained backbone (EfficientNet-B0, MobileNetV3-Small, ResNet18). Two-phase: Phase 1 — freeze backbone, train linear head (10–15 epochs, LR=1e-3). Phase 2 — unfreeze all, cosine LR schedule (backbone LR=1e-4, head LR=1e-3, 20–30 epochs).
- **10,000 ≤ N < 50,000** — lightweight to medium backbone (EfficientNet-B0/B2, ResNet34). Full fine-tuning or short 5-epoch frozen warm-up. Aggressive augmentation (random crop, flip, color jitter, random erasing).
- **N ≥ 50,000** — heavier backbone (EfficientNet-B4, ResNet50). Full fine-tuning with mixed precision (`torch.cuda.amp.autocast`).

Backbone selection:
- Load from `timm` (`timm.create_model(name, pretrained=True, num_classes=num_classes)`). Prefer small backbones for small N.
- For domain-specific images (medical, satellite, histology): check for domain-pretrained models (BiT, RadImageNet) — they transfer better than ImageNet weights.
- Do NOT use backbones >50M parameters with N<10k.
- Always use `native_h` and `native_w` from the skeleton in ALL transforms — never hardcode a resolution.

Resolution and channels:
- Use `transforms.Resize((native_h, native_w))` or `transforms.Resize(min(native_h, native_w))` + `CenterCrop`.
- If `img_channels == 1` (grayscale): add `transforms.Grayscale(num_output_channels=1)` and set the backbone's first conv `in_channels=1` (or use `Grayscale(num_output_channels=3)` with standard weights).
- Normalize with ImageNet mean/std for ImageNet-pretrained backbones: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

Augmentation (training only — val_transform must be resize + normalize only):
- Standard: `RandomHorizontalFlip`, `RandomCrop` with padding, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.
- For small N: also add `RandomRotation(15)` and `RandomErasing(p=0.25)`.

Class imbalance: use `WeightedRandomSampler` in the training DataLoader (weight = inverse class frequency) rather than SMOTE.
