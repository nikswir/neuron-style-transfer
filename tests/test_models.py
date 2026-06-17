"""Tests for the feature extractors.

These download pretrained weights, so they are gated behind ST_RUN_BACKBONE=1
(see conftest). Run with: ``ST_RUN_BACKBONE=1 poetry run pytest tests/test_models.py``.
"""

from __future__ import annotations

import torch
import pytest

from tests.conftest import requires_backbone
from style_transfer.models import build_extractor

########################################
#             Activations              #
########################################


@requires_backbone
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


@requires_backbone
@pytest.mark.parametrize("backbone", ["vgg16", "vgg19", "resnet50"])
def test_parameters_are_frozen(backbone):
    extractor = build_extractor(backbone)
    assert all(not p.requires_grad for p in extractor.parameters())
    assert not extractor.training


########################################
#             Layer counts             #
########################################


@requires_backbone
def test_vgg16_has_13_conv_layers():
    extractor = build_extractor("vgg16")
    conv_layers = [n for n in extractor.layer_names if n.startswith("conv")]
    assert len(conv_layers) == 13


@requires_backbone
def test_vgg19_has_16_conv_layers():
    extractor = build_extractor("vgg19")
    conv_layers = [n for n in extractor.layer_names if n.startswith("conv")]
    assert len(conv_layers) == 16


@requires_backbone
def test_resnet50_has_16_blocks():
    extractor = build_extractor("resnet50")
    assert len(extractor.layer_names) == 16


########################################
#                Errors                #
########################################


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_extractor("alexnet")
