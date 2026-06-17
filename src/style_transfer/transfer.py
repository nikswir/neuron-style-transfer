"""The style-transfer optimization loop.

The generated image starts as a clone of the content image and is treated as
the only trainable parameter. The (constant) content and style targets are
computed **once** up front; each optimizer step only re-runs the network on the
generated image. Both L-BFGS (default, as in Gatys et al.) and Adam are
supported.
"""

from __future__ import annotations

import torch

from torch import nn, optim
from collections.abc import Callable
from dataclasses import dataclass, field

from style_transfer import losses

# callback(step, image, losses_dict) -> None
StepCallback = Callable[[int, torch.Tensor, dict[str, float]], None]


########################################
#            Configuration             #
########################################


@dataclass
class TransferConfig:
    """Hyper-parameters for a single style-transfer run.

    ``content_layers`` / ``style_layers`` select which extractor outputs feed
    the corresponding loss. The per-layer weight dicts default to ``1.0`` for
    any selected layer not listed explicitly.
    """

    content_layers: list[str]
    style_layers: list[str]
    content_weights: dict[str, float] = field(default_factory=dict)
    style_weights: dict[str, float] = field(default_factory=dict)
    content_weight: float = 1.0
    style_weight: float = 1e6
    learning_rate: float = 1.0
    optimizer: str = "lbfgs"
    tv_weight: float = 1.0
    steps: int = 50

    def weight_for(
        self,
        layer: str,
        kind: str,
    ) -> float:
        table = self.content_weights if kind == "content" else self.style_weights
        return table.get(layer, 1.0)


########################################
#                Result                #
########################################


@dataclass
class TransferResult:
    """Output of :func:`run_style_transfer`."""

    image: torch.Tensor
    history: dict[str, list[float]]


########################################
#          Optimizer factory           #
########################################


def _make_optimizer(
    cfg: TransferConfig,
    image: torch.Tensor,
) -> optim.Optimizer:
    if cfg.optimizer == "lbfgs":
        return optim.LBFGS([image], lr=cfg.learning_rate, max_iter=20)
    if cfg.optimizer == "adam":
        return optim.Adam([image], lr=cfg.learning_rate)
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


########################################
#            Target capture            #
########################################


@torch.no_grad()
def _capture(
    extractor: nn.Module,
    image: torch.Tensor,
    layers: list[str],
) -> dict[str, torch.Tensor]:
    feats = extractor(image)
    return {name: feats[name].detach() for name in layers}


########################################
#          Optimization loop           #
########################################


def run_style_transfer(
    content: torch.Tensor,
    style: torch.Tensor,
    extractor: nn.Module,
    cfg: TransferConfig,
    *,
    callback: StepCallback | None = None,
) -> TransferResult:
    """Run image-optimization style transfer.

    Parameters
    ----------
    content, style:
        Normalized ``(1, 3, H, W)`` tensors on the target device.
    extractor:
        A frozen feature extractor from :mod:`style_transfer.models`.
    cfg:
        Hyper-parameters.
    callback:
        Optional ``callback(step, image, losses_dict)`` invoked once per step.
    """
    # ── 1. The generated image is the only trainable tensor ──
    generated = content.clone().requires_grad_(True)

    # ── 2. Pre-compute the constant content / style targets ──
    content_targets = _capture(extractor, content, cfg.content_layers)
    style_targets = _capture(extractor, style, cfg.style_layers)

    # ── 3. Optimizer and per-term loss history ──
    optimizer = _make_optimizer(cfg, generated)
    history: dict[str, list[float]] = {
        "content": [],
        "style": [],
        "tv": [],
        "total": [],
    }

    def closure() -> torch.Tensor:
        # ── Re-run the network on the current image ──
        optimizer.zero_grad()
        feats = extractor(generated)

        # ── Content term ──
        c_loss = generated.new_zeros(())
        for layer in cfg.content_layers:
            c_loss = c_loss + cfg.weight_for(layer, "content") * losses.content_loss(
                feats[layer], content_targets[layer]
            )

        # ── Style term ──
        s_loss = generated.new_zeros(())
        for layer in cfg.style_layers:
            s_loss = s_loss + cfg.weight_for(layer, "style") * losses.style_loss(
                feats[layer], style_targets[layer]
            )

        # ── Total-variation term, then the weighted sum ──
        tv = losses.total_variation_loss(generated)
        total = cfg.content_weight * c_loss + cfg.style_weight * s_loss + cfg.tv_weight * tv
        total.backward()

        # ── Record the per-term history ──
        history["content"].append(float(c_loss.detach()))
        history["style"].append(float(s_loss.detach()))
        history["tv"].append(float(tv.detach()))
        history["total"].append(float(total.detach()))
        return total

    # ── 4. Step the optimizer, invoking the callback per step ──
    for i in range(cfg.steps):
        # torch stubs type the closure as `() -> float`, but L-BFGS expects it
        # to return the loss tensor (which is what we do); the stub is imprecise.
        optimizer.step(closure)  # type: ignore[arg-type]
        if callback is not None:
            with torch.no_grad():
                last = {k: v[-1] for k, v in history.items() if v}
            callback(i, generated.detach(), last)

    return TransferResult(image=generated.detach(), history=history)
