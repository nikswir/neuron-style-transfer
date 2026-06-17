"""Feature extractors that expose intermediate activations.

Style transfer needs the responses of several intermediate layers, not the
final classification output. Each extractor wraps a pretrained torchvision
backbone and returns an ``OrderedDict`` mapping a human-readable layer name to
its activation tensor. All backbone parameters are frozen and the modules are
kept in ``eval`` mode -- only the generated image is ever optimized.
"""

from __future__ import annotations

import torch

from torch import nn
from torchvision import models
from collections import OrderedDict

########################################
#            Freeze helper             #
########################################


def _freeze(
    module: nn.Module,
) -> nn.Module:
    """Disable grads on every parameter and switch the module to ``eval``."""
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


########################################
#        VGG feature extractor         #
########################################


class VGGFeatures(nn.Module):
    """Expose the activation after every conv+ReLU pair of a VGG backbone.

    Layers are named ``conv{block}_{idx}`` following the convention of the
    original paper, where ``block`` increments after each max-pool. This works
    for both VGG16 and VGG19 since they share the same block structure.
    """

    def __init__(
        self,
        backbone: str = "vgg16",
        weights: str | None = "DEFAULT",
    ):
        super().__init__()

        # ── 1. Pick the backbone ────────────────────
        if backbone == "vgg16":
            net = models.vgg16(weights=weights)
        elif backbone == "vgg19":
            net = models.vgg19(weights=weights)
        else:
            raise ValueError(f"unknown VGG backbone: {backbone!r}")

        # ── 2. Split features into conv+ReLU / pool blocks ──
        self.blocks = nn.ModuleList()
        self.layer_names: list[str] = []
        block, idx, current = 1, 0, []
        for layer in net.features:
            current.append(layer)
            if isinstance(layer, nn.ReLU):
                # ReLU is made non-inplace so stored activations survive.
                current[-1] = nn.ReLU(inplace=False)
                idx += 1
                self.blocks.append(nn.Sequential(*current))
                self.layer_names.append(f"conv{block}_{idx}")
                current = []
            elif isinstance(layer, nn.MaxPool2d):
                self.blocks.append(nn.Sequential(*current))
                self.layer_names.append(f"pool{block}")
                block, idx, current = block + 1, 0, []

        # ── 3. Freeze every parameter ───────────────
        _freeze(self)

    def forward(
        self,
        x: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        # ── Run each block, recording its output ──
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, blk in zip(self.layer_names, self.blocks, strict=True):
            x = blk(x)
            out[name] = x
        return out


########################################
#       ResNet feature extractor       #
########################################


class ResNetFeatures(nn.Module):
    """Expose the activation after every bottleneck block of ResNet50.

    The stem (conv1+bn+relu+maxpool) is run once, then each block of the four
    residual stages is applied sequentially; the output of every block is
    recorded as ``stage{s}_block{b}``. torchvision's ``Bottleneck`` already
    implements the residual connection, so no manual re-wiring is needed.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        weights: str | None = "DEFAULT",
    ):
        super().__init__()

        # ── 1. Pick the backbone and isolate its stem ──
        if backbone != "resnet50":
            raise ValueError(f"unknown ResNet backbone: {backbone!r}")
        net = models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)

        # ── 2. Index every bottleneck block by stage ──
        self.blocks = nn.ModuleList()
        self.layer_names: list[str] = []
        stages = [net.layer1, net.layer2, net.layer3, net.layer4]
        for stage_idx, stage in enumerate(stages, start=1):
            for block_idx, block in enumerate(stage):
                self.blocks.append(block)
                self.layer_names.append(f"stage{stage_idx}_block{block_idx}")

        # ── 3. Freeze every parameter ───────────────
        _freeze(self)

    def forward(
        self,
        x: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        # ── Stem once, then each block in turn ──
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        x = self.stem(x)
        for name, blk in zip(self.layer_names, self.blocks, strict=True):
            x = blk(x)
            out[name] = x
        return out


########################################
#          Extractor factory           #
########################################


def build_extractor(
    backbone: str,
    weights: str | None = "DEFAULT",
) -> nn.Module:
    """Factory: return a frozen feature extractor for the given backbone."""
    # ── Dispatch on the backbone name ──
    backbone = backbone.lower()
    if backbone in ("vgg16", "vgg19"):
        return VGGFeatures(backbone, weights=weights)
    if backbone == "resnet50":
        return ResNetFeatures(backbone, weights=weights)
    raise ValueError(f"unsupported backbone: {backbone!r}")
