"""Tests for the optimization loop with a tiny fake extractor (no downloads)."""

from __future__ import annotations

import torch
import pytest

from torch import nn
from collections import OrderedDict

from style_transfer.transfer import (
    TransferConfig,
    _make_optimizer,
    run_style_transfer,
)

########################################
#             Test doubles             #
########################################


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
    base = {
        "content_layers": ["b"],
        "style_layers": ["a", "b"],
        "steps": 3,
        "optimizer": "adam",
        "learning_rate": 0.1,
        "style_weight": 1e3,
    }
    base.update(kw)
    return TransferConfig(**base)


########################################
#          Optimization loop           #
########################################


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
    images = []
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)

    losses_seen = []

    def record(i, img, losses):
        calls.append(i)
        images.append(img)
        losses_seen.append(losses)

    run_style_transfer(
        content,
        style,
        TinyExtractor(),
        _cfg(steps=3),
        callback=record,
    )
    assert calls == [0, 1, 2]

    # The callback must receive the live generated image (detached), not a
    # placeholder -- pins the image argument passed at the call site.
    assert all(img is not None for img in images)
    assert all(img.shape == content.shape for img in images)
    assert all(not img.requires_grad for img in images)

    # The third argument is the per-step loss snapshot: every recorded term,
    # finite. Pins the `last` mapping -- a mutant nulling it would surface here.
    assert all(
        set(d) == {"content", "style", "tv", "total"} for d in losses_seen
    )
    assert all(
        all(isinstance(v, float) for v in d.values()) for d in losses_seen
    )


def test_lbfgs_optimizer_runs():
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    result = run_style_transfer(
        content,
        style,
        TinyExtractor(),
        _cfg(optimizer="lbfgs", learning_rate=0.5, steps=2),
    )
    assert torch.isfinite(result.image).all()


########################################
#           Loss composition           #
########################################


def test_total_is_weighted_sum_of_terms_and_terms_nonnegative():
    # Use distinct global weights so the composition is sensitive to each term
    # and to its sign / operator. Each recorded `total` must equal
    # content_weight*content + style_weight*style + tv_weight*tv, and every
    # individual term (an MSE) must be non-negative.
    torch.manual_seed(0)
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    cfg = _cfg(content_weight=2.0, style_weight=3.0, tv_weight=5.0, steps=4)
    h = run_style_transfer(content, style, TinyExtractor(), cfg).history
    for c, s, tv, total in zip(
        h["content"],
        h["style"],
        h["tv"],
        h["total"],
        strict=True,
    ):
        assert c >= 0.0
        assert s >= 0.0
        assert tv >= 0.0
        assert total == pytest.approx(2.0 * c + 3.0 * s + 5.0 * tv, rel=1e-5)
    # After at least one optimizer step the content term must actually be > 0,
    # so a sign flip inside the content accumulation would surface as negative.
    assert max(h["content"]) > 0.0
    assert max(h["style"]) > 0.0


def test_content_loop_reads_content_weight_table():
    # The content loop looks up content_weights, the style loop style_weights
    # (weight_for routes on the "content"/"style" kind). Zero layer "b" in
    # content_weights but give it a large value in style_weights: with correct
    # routing the content term stays exactly 0, while the non-zero style weight
    # drives the image away from the content so an *unweighted* content loss is
    # > 0. If the content loop ever read the style table (a wrong/typo'd kind,
    # which falls through to style_weights), the content term would jump
    # above 0. This pins the kind argument, not just the layer key.
    torch.manual_seed(0)
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    cfg = _cfg(
        steps=3,
        content_weights={"b": 0.0},
        style_weights={"a": 0.5, "b": 7.0},
    )
    h = run_style_transfer(content, style, TinyExtractor(), cfg).history
    assert all(
        c == 0.0 for c in h["content"]
    )  # content term genuinely silenced
    assert max(h["style"]) > 0.0  # ...while the image really did move


def test_style_loop_zero_weights_silence_style_term():
    # Zeroing every per-layer style weight drives the style term to exactly 0.
    # This pins the *layer key* in the style loop: a mutant that looks up a
    # constant key (e.g. None) falls back to 1.0 and leaves the term non-zero.
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    cfg = _cfg(steps=2, style_weights={"a": 0.0, "b": 0.0})
    h = run_style_transfer(content, style, TinyExtractor(), cfg).history
    assert all(s == 0.0 for s in h["style"])


def test_first_step_losses_match_analytic_sum():
    # At the first closure `generated` is still a clone of `content`, so the
    # recorded terms must equal the plain (unit-weighted) sum of the per-layer
    # losses computed directly. Pins the accumulation operator: turning the
    # `weight * loss` product into a division would change these values.
    from style_transfer import losses as L

    torch.manual_seed(0)
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16)
    ext = TinyExtractor()
    cfg = _cfg(steps=1)
    h = run_style_transfer(content, style, ext, cfg).history
    with torch.no_grad():
        cf, sf = ext(content), ext(style)
        exp_c = sum(
            float(L.content_loss(cf[layer], cf[layer]))
            for layer in cfg.content_layers
        )
        exp_s = sum(
            float(L.style_loss(cf[layer], sf[layer]))
            for layer in cfg.style_layers
        )
    assert h["content"][0] == pytest.approx(exp_c, abs=1e-7)
    assert h["style"][0] == pytest.approx(exp_s, rel=1e-5)


########################################
#            Weight routing            #
########################################


def test_weight_for_routes_on_kind():
    # weight_for sends kind "content" to content_weights and everything else to
    # style_weights. Give the same layer key opposite weights in the two tables
    # so a swapped table or a mutated "content" literal flips the result; an
    # absent layer must fall back to the 1.0 default in either table.
    cfg = _cfg(content_weights={"a": 9.0}, style_weights={"a": 0.0})
    assert cfg.weight_for("a", "content") == 9.0
    assert cfg.weight_for("a", "style") == 0.0
    assert cfg.weight_for("missing", "content") == 1.0
    assert cfg.weight_for("missing", "style") == 1.0


########################################
#          Optimizer factory           #
########################################


def test_make_optimizer_passes_learning_rate():
    # A real bug would be silently optimizing at the wrong rate: assert the
    # configured learning rate actually reaches the optimizer for both backends.
    image = torch.zeros(1, 3, 4, 4, requires_grad=True)
    adam = _make_optimizer(_cfg(optimizer="adam", learning_rate=0.123), image)
    lbfgs = _make_optimizer(_cfg(optimizer="lbfgs", learning_rate=0.456), image)
    assert adam.param_groups[0]["lr"] == 0.123
    assert lbfgs.param_groups[0]["lr"] == 0.456
    # L-BFGS runs a fixed inner loop; pin max_iter so dropping or changing it
    # (a quietly different optimizer) is caught.
    assert lbfgs.param_groups[0]["max_iter"] == 20


def test_make_optimizer_rejects_unknown():
    image = torch.zeros(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(ValueError, match="unknown optimizer"):
        _make_optimizer(_cfg(optimizer="rmsprop"), image)
