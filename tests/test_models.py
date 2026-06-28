"""Tests for the feature extractors.

These download pretrained weights, so they are gated as stage 2 (see
conftest). Run with:
``RUN_STAGE2=1 uv run pytest tests/test_models.py``.
"""

from __future__ import annotations

import torch
import pytest

from torch import nn

from tests.conftest import stage2
from style_transfer import models
from style_transfer.models import build_extractor

########################################
#             Activations              #
########################################


@stage2
@pytest.mark.parametrize("backbone", ["vgg16", "vgg19", "resnet50"])
def test_forward_returns_named_activations(backbone):
    extractor = build_extractor(backbone)
    out = extractor(torch.randn(1, 3, 64, 64))
    assert set(out) == set(extractor.layer_names)
    assert len(out) > 0
    for tensor in out.values():
        assert tensor.dim() == 4


########################################
#               Freezing               #
########################################


@stage2
@pytest.mark.parametrize("backbone", ["vgg16", "vgg19", "resnet50"])
def test_parameters_are_frozen(backbone):
    extractor = build_extractor(backbone)
    assert all(not p.requires_grad for p in extractor.parameters())
    assert not extractor.training


########################################
#             Layer counts             #
########################################


@stage2
def test_vgg16_has_13_conv_layers():
    extractor = build_extractor("vgg16")
    conv_layers = [n for n in extractor.layer_names if n.startswith("conv")]
    assert len(conv_layers) == 13


@stage2
def test_vgg19_has_16_conv_layers():
    extractor = build_extractor("vgg19")
    conv_layers = [n for n in extractor.layer_names if n.startswith("conv")]
    assert len(conv_layers) == 16


@stage2
def test_resnet50_has_16_blocks():
    extractor = build_extractor("resnet50")
    assert len(extractor.layer_names) == 16


########################################
#        Block naming (offline)        #
########################################

# The conv/relu/pool split and the conv{b}_{i} / pool{b} / stage{s}_block{b}
# naming is pure logic, but it lives inside __init__ behind a weights download
# (stage 2). Feed a stub backbone so the naming is exercised on every run.


def test_vgg_block_naming_without_weights(monkeypatch):
    fake = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.ReLU(),
        nn.Conv2d(3, 3, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(3, 3, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    class _FakeVGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = fake

    monkeypatch.setattr(
        models.models,
        "vgg16",
        lambda weights=None: _FakeVGG(),
    )

    extractor = models.VGGFeatures("vgg16", weights=None)
    assert extractor.layer_names == [
        "conv1_1",
        "conv1_2",
        "pool1",
        "conv2_1",
        "pool2",
    ]


def test_resnet_block_naming_without_weights(monkeypatch):
    class _FakeResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, 3)
            self.bn1 = nn.BatchNorm2d(3)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2)
            self.layer1 = nn.Sequential(nn.Identity(), nn.Identity())
            self.layer2 = nn.Sequential(nn.Identity())
            self.layer3 = nn.Sequential(nn.Identity())
            self.layer4 = nn.Sequential(nn.Identity())

    monkeypatch.setattr(
        models.models,
        "resnet50",
        lambda weights=None: _FakeResNet(),
    )

    extractor = models.ResNetFeatures("resnet50", weights=None)
    assert extractor.layer_names == [
        "stage1_block0",
        "stage1_block1",
        "stage2_block0",
        "stage3_block0",
        "stage4_block0",
    ]


########################################
#                Errors                #
########################################


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_extractor("alexnet")
