"""Property-based checks for the loss functions."""

from __future__ import annotations

import torch

from style_transfer import losses

########################################
#             Content loss             #
########################################


def test_content_loss_zero_on_identity():
    x = torch.randn(1, 4, 8, 8)
    assert torch.isclose(losses.content_loss(x, x), torch.tensor(0.0))


########################################
#             Gram matrix              #
########################################


def test_gram_matrix_shape_and_symmetry():
    x = torch.randn(2, 5, 8, 8)
    g = losses.gram_matrix(x)
    assert g.shape == (2, 5, 5)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-5)


def test_gram_matrix_positive_semidefinite():
    x = torch.randn(1, 6, 8, 8)
    g = losses.gram_matrix(x)[0]
    eigvals = torch.linalg.eigvalsh(g)
    assert (eigvals >= -1e-5).all()


def test_gram_matrix_invariant_to_spatial_permutation():
    # The defining property of the style representation: the Gram matrix
    # encodes channel correlations, not spatial layout. Shuffling the spatial
    # positions of a feature map must leave the Gram matrix unchanged.
    x = torch.randn(1, 5, 8, 8)
    flat = x.view(1, 5, -1)
    perm = torch.randperm(flat.shape[-1])
    shuffled = flat[:, :, perm].view(1, 5, 8, 8)
    assert torch.allclose(
        losses.gram_matrix(x),
        losses.gram_matrix(shuffled),
        atol=1e-5,
    )


def test_gram_matrix_normalization_factor():
    # Pins the C*H*W divisor: for an all-ones (B, C, H, W) map every Gram entry
    # is H*W before normalization, so the normalized value must be exactly 1/C.
    # Catches mutations of the c * h * w divisor (e.g. c / h * w, c * h / w).
    c, h, w = 5, 4, 8
    g = losses.gram_matrix(torch.ones(1, c, h, w))
    assert torch.allclose(g, torch.full((1, c, c), 1.0 / c), atol=1e-6)


########################################
#              Style loss              #
########################################


def test_style_loss_zero_on_identity():
    x = torch.randn(1, 4, 8, 8)
    assert torch.isclose(losses.style_loss(x, x), torch.tensor(0.0), atol=1e-6)


########################################
#         Total-variation loss         #
########################################


def test_tv_loss_zero_on_constant_image():
    flat = torch.full((1, 3, 16, 16), 0.5)
    assert torch.isclose(losses.total_variation_loss(flat), torch.tensor(0.0))


def test_tv_loss_uses_spatial_axes_not_channels():
    # An image constant across H and W but varying across channels must have
    # zero TV. This is the regression test for the original axis bug.
    img = torch.zeros(1, 3, 16, 16)
    img[:, 1] = 1.0
    img[:, 2] = 2.0
    assert torch.isclose(losses.total_variation_loss(img), torch.tensor(0.0))


def test_tv_loss_positive_on_spatial_variation():
    img = torch.zeros(1, 3, 16, 16)
    img[:, :, ::2, :] = 1.0  # alternating rows -> spatial variation
    assert losses.total_variation_loss(img).item() > 0


def test_tv_loss_exact_value_pins_both_axes():
    # A 3x3 image with a distinct value in every cell: both the height and width
    # neighbour-difference slices then have length > 1, so the slice *direction*
    # matters. (A 2x2 image cannot pin this -- there `:-1` and `:1` select the
    # same single row/column, leaving `:-1 -> :+1` mutations alive.) Comparing to
    # F.mse_loss on the correct slices also pins the diff_h + diff_w combination.
    img = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 5.0, 8.0], [13.0, 21.0, 34.0]]]])
    expected_h = torch.nn.functional.mse_loss(img[:, :, :-1, :], img[:, :, 1:, :])
    expected_w = torch.nn.functional.mse_loss(img[:, :, :, :-1], img[:, :, :, 1:])
    tv = losses.total_variation_loss(img)
    assert torch.isclose(tv, expected_h + expected_w)
    assert tv.item() > 0
