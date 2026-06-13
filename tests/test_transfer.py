"""Tests for the optimization loop, using a tiny fake extractor (no downloads)."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from style_transfer.transfer import TransferConfig, run_style_transfer


class TinyExtractor(nn.Module):
    """A frozen 2-conv stand-in for a real backbone."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 4, 3, padding=1)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        a = torch.relu(self.c1(x))
        b = torch.relu(self.c2(a))
        return OrderedDict([("a", a), ("b", b)])


def _cfg(**kw):
    base = dict(
        content_layers=["b"],
        style_layers=["a", "b"],
        steps=3,
        optimizer="adam",
        learning_rate=0.1,
        style_weight=1e3,
    )
    base.update(kw)
    return TransferConfig(**base)


def test_run_returns_same_shape():
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    result = run_style_transfer(content, style, TinyExtractor(), _cfg())
    assert result.image.shape == content.shape
    assert not result.image.requires_grad


def test_history_is_recorded_and_finite():
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    result = run_style_transfer(content, style, TinyExtractor(), _cfg(steps=4))
    assert len(result.history["total"]) >= 4
    assert all(torch.isfinite(torch.tensor(v)) for v in result.history["total"])


def test_optimization_reduces_total_loss():
    torch.manual_seed(0)
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    result = run_style_transfer(content, style, TinyExtractor(), _cfg(steps=15))
    assert result.history["total"][-1] < result.history["total"][0]


def test_callback_invoked_per_step():
    calls = []
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    run_style_transfer(
        content,
        style,
        TinyExtractor(),
        _cfg(steps=3),
        callback=lambda i, img, losses: calls.append(i),
    )
    assert calls == [0, 1, 2]


def test_lbfgs_optimizer_runs():
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    result = run_style_transfer(
        content, style, TinyExtractor(), _cfg(optimizer="lbfgs", learning_rate=0.5, steps=2)
    )
    assert torch.isfinite(result.image).all()
