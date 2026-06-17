"""Shared pytest fixtures. Tests run on CPU with tiny tensors by default."""

from __future__ import annotations

import os
import torch
import pytest

########################################
#          Backbone test gate          #
########################################

# Backbone tests download pretrained weights; skip them unless explicitly
# enabled (set ST_RUN_BACKBONE=1) so the core suite stays offline and fast.
RUN_BACKBONE = os.environ.get("ST_RUN_BACKBONE") == "1"

requires_backbone = pytest.mark.skipif(
    not RUN_BACKBONE,
    reason="set ST_RUN_BACKBONE=1 to run tests that download pretrained weights",
)


########################################
#               Fixtures               #
########################################


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def small_image() -> torch.Tensor:
    """A deterministic small normalized image batch ``(1, 3, 16, 16)``."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 3, 16, 16, generator=g)
