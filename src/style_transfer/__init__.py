"""Neural style transfer by image optimization (Gatys et al., 2016).

Compares VGG16, VGG19 and ResNet50 backbones as feature extractors.
"""

from __future__ import annotations

from style_transfer import losses, models, transfer, transforms
from style_transfer.models import VGGFeatures, ResNetFeatures, build_extractor
from style_transfer.transforms import load_image, save_image, build_transforms
from style_transfer.transfer import TransferConfig, TransferResult, run_style_transfer

__version__ = "0.1.0"

# Grouped by kind, each group laddered by length.
__all__ = [
    # ── submodules ──
    "losses",
    "models",
    "transfer",
    "transforms",
    # ── classes ──
    "VGGFeatures",
    "ResNetFeatures",
    "TransferConfig",
    "TransferResult",
    # ── functions ──
    "load_image",
    "save_image",
    "build_extractor",
    "build_transforms",
    "run_style_transfer",
]
