"""
Brain Tumor MRI Classification with EfficientNet-B4

A multi-phase image classification training pipeline using:
- EfficientNet-B4 from timm (transfer learning on ImageNet)
- Brain Tumor MRI Dataset from Kaggle (4 classes, ~7,000 MRI slices)
- Focal loss with class weights for imbalanced data
- Mixup augmentation and gradient clipping for training stability
- Union/Flyte for orchestration with Union UI report panels
"""

__version__ = "1.0.0"
