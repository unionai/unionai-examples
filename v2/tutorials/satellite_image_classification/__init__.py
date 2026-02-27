"""
Satellite Image Classification with EfficientNet-B0 on EuroSAT

A multi-phase image classification training pipeline using:
- EfficientNet-B0 from timm (transfer learning on ImageNet)
- EuroSAT satellite imagery dataset from HuggingFace
- Weights & Biases for metrics and 3D t-SNE visualization
- Union/Flyte for orchestration
"""

__version__ = "1.0.0"
