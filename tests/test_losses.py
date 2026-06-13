"""Property-based checks for the loss functions."""

from __future__ import annotations

import torch

from style_transfer import losses


def test_content_loss_zero_on_identity():
    x = torch.randn(1, 4, 8, 8)
    assert torch.isclose(losses.content_loss(x, x), torch.tensor(0.0))


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


def test_style_loss_zero_on_identity():
    x = torch.randn(1, 4, 8, 8)
    assert torch.isclose(losses.style_loss(x, x), torch.tensor(0.0), atol=1e-6)


def test_gram_matrix_invariant_to_spatial_permutation():
    # The defining property of the style representation: the Gram matrix
    # encodes channel correlations, not spatial layout. Shuffling the spatial
    # positions of a feature map must leave the Gram matrix unchanged.
    x = torch.randn(1, 5, 8, 8)
    flat = x.view(1, 5, -1)
    perm = torch.randperm(flat.shape[-1])
    shuffled = flat[:, :, perm].view(1, 5, 8, 8)
    assert torch.allclose(losses.gram_matrix(x), losses.gram_matrix(shuffled), atol=1e-5)


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
