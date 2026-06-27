"""Shared pytest fixtures. Tests run on CPU with tiny tensors by default."""

from __future__ import annotations

import os
import torch
import pytest

########################################
#          Stage-2 test gate           #
########################################

# Stage-2 tests are heavy (download data/weights, GPU, slow); skip them unless
# explicitly enabled (RUN_STAGE2=1) so the default suite stays offline and fast.
RUN_STAGE2 = os.environ.get("RUN_STAGE2") == "1"

stage2 = pytest.mark.skipif(
    not RUN_STAGE2,
    reason="set RUN_STAGE2=1 to run heavy stage-2 tests",
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
