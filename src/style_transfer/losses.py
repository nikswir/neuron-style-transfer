"""Loss functions for neural style transfer (Gatys et al., 2016).

Three terms are combined during optimization:

* **content loss** -- MSE between the feature maps of the generated and the
  content image at a chosen deep layer; preserves *what* is depicted.
* **style loss** -- MSE between the Gram matrices of the feature maps; the Gram
  matrix captures channel correlations, i.e. texture and color statistics,
  independent of spatial layout.
* **total-variation loss** -- a smoothness regularizer penalizing differences
  between spatially neighbouring pixels; suppresses high-frequency noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

########################################
#             Content loss             #
########################################


def content_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between two feature maps of equal shape."""
    return F.mse_loss(generated, target)


########################################
#         Style representation         #
########################################


def gram_matrix(
    features: torch.Tensor,
) -> torch.Tensor:
    """Normalized Gram matrix of a feature map batch ``(B, C, H, W)``.

    Returns a ``(B, C, C)`` tensor divided by ``C * H * W`` so its magnitude
    does not scale with feature-map *size*. This does not, however, equalize
    the style term across layers: :func:`style_loss` averages over the
    ``C * C`` entries, so a layer's contribution still scales with its channel
    count ``C``.
    """
    # ── Channel correlations over all spatial positions ──
    b, c, h, w = features.size()
    flat = features.view(b, c, h * w)
    gram = torch.bmm(flat, flat.transpose(1, 2))

    # ── Normalize so the magnitude is size-independent ──
    return gram.div(c * h * w)


def style_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between the Gram matrices of two feature maps."""
    return F.mse_loss(gram_matrix(generated), gram_matrix(target))


########################################
#         Total-variation loss         #
########################################


def total_variation_loss(
    image: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic total-variation regularizer on a ``(B, C, H, W)`` image.

    The height (dim 2) and width (dim 3) neighbour differences are penalized
    separately and summed -- the *anisotropic* form, not combined under a
    square root -- and *not* taken across colour channels.
    """
    # ── Neighbour differences along height, then width ──
    diff_h = F.mse_loss(image[:, :, :-1, :], image[:, :, 1:, :])
    diff_w = F.mse_loss(image[:, :, :, :-1], image[:, :, :, 1:])
    return diff_h + diff_w
