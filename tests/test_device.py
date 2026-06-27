"""Tests for device selection."""

from __future__ import annotations

import torch

from style_transfer.device import pick_device

########################################
#           Device selection           #
########################################


def test_pick_device_explicit_cpu():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_resolves_without_raising():
    # "auto" must resolve to a usable device without raising, preferring
    # CUDA -> MPS -> CPU. This pins the resolution order and every device
    # literal: mutating any of them makes "auto" raise or pick the wrong one.
    expected = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    assert pick_device("auto").type == expected


def test_pick_device_auto_prefers_cuda(monkeypatch):
    # Force CUDA available: "auto" must pick it ahead of MPS / CPU. Pins the
    # "cuda" literal and the first branch independently of the host hardware.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device("auto").type == "cuda"


def test_pick_device_auto_falls_back_to_mps(monkeypatch):
    # No CUDA but MPS available: "auto" must pick MPS. Pins the "mps" literal,
    # the `mps is not None` guard, and the CUDA-before-MPS ordering.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("auto").type == "mps"


def test_pick_device_auto_falls_back_to_cpu(monkeypatch):
    # Neither accelerator available: "auto" must fall through to CPU. Pins the
    # final "cpu" literal and that an unavailable MPS does not get selected.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto").type == "cpu"
