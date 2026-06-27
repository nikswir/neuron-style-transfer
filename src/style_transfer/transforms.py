"""Image <-> tensor conversions for neural style transfer.

The backbone networks (VGG/ResNet) are pretrained on ImageNet, so inputs must
be normalized with the ImageNet channel statistics. ``to_tensor`` produces a
4D batch tensor ``(1, 3, H, W)`` ready for the network; ``to_image`` is the
exact inverse (de-normalize -> clamp -> PIL image) used to visualize results.
"""

from __future__ import annotations

import torch

from PIL import Image
from pathlib import Path
from torchvision import transforms
from collections.abc import Callable

ToTensor = Callable[[Image.Image], torch.Tensor]
ToImage = Callable[[torch.Tensor], Image.Image]


########################################
#         ImageNet statistics          #
########################################

# ImageNet statistics used by all torchvision pretrained models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


########################################
#          Transform builders          #
########################################


def build_transforms(
    image_size: int = 512,
) -> tuple[ToTensor, ToImage]:
    """Return a ``(to_tensor, to_image)`` pair of callables.

    ``to_tensor(PIL.Image) -> Tensor[1, 3, S, S]`` resizes to a square,
    converts to a tensor, normalizes with ImageNet stats and adds a batch
    dimension. ``to_image(Tensor) -> PIL.Image`` inverts the normalization,
    clamps to ``[0, 1]`` and returns a viewable image.
    """
    # ── Inverse normalization parameters ──
    inv_mean = [
        -m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
    ]
    inv_std = [1.0 / s for s in IMAGENET_STD]

    # ── Forward: PIL image -> normalized batch tensor ──
    to_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ],
    )

    # ── Inverse: tensor -> clamped, viewable PIL image ──
    to_image = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.detach().cpu().squeeze(0)),
            transforms.Normalize(mean=inv_mean, std=inv_std),
            transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0)),
            transforms.ToPILImage(),
        ],
    )
    return to_tensor, to_image


########################################
#               File I/O               #
########################################


def load_image(
    path: str | Path,
    image_size: int = 512,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load an image file and return its normalized ``(1, 3, S, S)`` tensor."""
    to_tensor, _ = build_transforms(image_size)
    image = Image.open(path).convert("RGB")
    return to_tensor(image).to(device)


def save_image(
    tensor: torch.Tensor,
    path: str | Path,
) -> None:
    """De-normalize a tensor and write it to ``path`` as an image file.

    The output size is the tensor's own spatial size; the inverse transform
    only de-normalizes and clamps, so no ``image_size`` is needed here.
    """
    _, to_image = build_transforms()
    to_image(tensor).save(path)
