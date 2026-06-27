"""Property-based tests (Hypothesis).

Where the example-based tests pin one hand-picked input, these assert
*invariants* that must hold for every input Hypothesis can generate: an MSE is
never negative, ``f(x, x)`` is zero, a Gram matrix is symmetric, the
total-variation of a flat image vanishes, and the tensor<->image conversion is
a round-trip. Tensors are kept tiny on purpose so 100 generated examples stay
fast.
"""

from __future__ import annotations

import torch
import pytest
import numpy as np

from PIL import Image
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from style_transfer import losses
from style_transfer.transforms import build_transforms

########################################
#              Strategies              #
########################################

# A small (B, C, H, W) feature map of finite float32 values.
feature_map = arrays(
    dtype=np.float32,
    shape=(1, 3, 4, 4),
    elements=st.floats(
        -10.0,
        10.0,
        allow_infinity=False,
        allow_nan=False,
        width=32,
    ),
)


########################################
#           Loss invariants            #
########################################


@given(feature_map)
def test_content_loss_is_zero_on_identity(x: np.ndarray) -> None:
    t = torch.from_numpy(x)
    assert losses.content_loss(t, t).item() == 0.0


@given(feature_map, feature_map)
def test_content_loss_is_nonnegative_and_symmetric(
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    a, b = torch.from_numpy(x), torch.from_numpy(y)
    loss_forward = losses.content_loss(a, b).item()
    loss_swapped = losses.content_loss(b, a).item()
    assert loss_forward >= 0.0
    assert loss_forward == pytest.approx(loss_swapped)


@given(feature_map, feature_map)
def test_style_loss_is_nonnegative_and_zero_on_identity(
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    a, b = torch.from_numpy(x), torch.from_numpy(y)
    assert losses.style_loss(a, b).item() >= 0.0
    assert losses.style_loss(a, a).item() == pytest.approx(0.0, abs=1e-6)


@given(feature_map)
def test_gram_matrix_is_symmetric(x: np.ndarray) -> None:
    g = losses.gram_matrix(torch.from_numpy(x))
    assert g.shape == (1, 3, 3)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-5)


@given(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32))
def test_tv_loss_vanishes_on_constant_image(value: float) -> None:
    image = torch.full((1, 3, 8, 8), value)
    assert losses.total_variation_loss(image).item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


########################################
#         Transform round-trip         #
########################################


@given(arrays(dtype=np.uint8, shape=(16, 16, 3), elements=st.integers(0, 255)))
@settings(max_examples=50)
def test_transforms_round_trip_preserves_pixels(pixels: np.ndarray) -> None:
    # Image is already the target size, so resize is a no-op; the only lossy
    # step is the uint8 quantization, which round-trips to within one level.
    to_tensor, to_image = build_transforms(16)
    original = Image.fromarray(pixels, mode="RGB")
    recovered = np.asarray(to_image(to_tensor(original)))
    assert np.abs(recovered.astype(int) - pixels.astype(int)).max() <= 1
