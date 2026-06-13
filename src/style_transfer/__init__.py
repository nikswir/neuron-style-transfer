"""Neural style transfer by image optimization (Gatys et al., 2016).

Compares VGG16, VGG19 and ResNet50 backbones as feature extractors.
"""

from __future__ import annotations

from . import losses, models, transfer, transforms
from .models import ResNetFeatures, VGGFeatures, build_extractor
from .transfer import TransferConfig, TransferResult, run_style_transfer
from .transforms import build_transforms, load_image, save_image

__version__ = "0.1.0"

__all__ = [
    "losses",
    "models",
    "transfer",
    "transforms",
    "build_extractor",
    "VGGFeatures",
    "ResNetFeatures",
    "TransferConfig",
    "TransferResult",
    "run_style_transfer",
    "build_transforms",
    "load_image",
    "save_image",
]
