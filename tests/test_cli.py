"""Integration tests for the command-line entry point.

The backbone is replaced with a tiny stand-in (TinyExtractor) so the CLI wiring
-- argument parsing, pairing, image loading, the optimization loop and writing
the output files -- is exercised end-to-end without downloading pretrained
weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from style_transfer import cli

from .test_transfer import TinyExtractor


def test_pick_device_explicit_cpu():
    assert cli.pick_device("cpu") == torch.device("cpu")


def test_pair_inputs_by_position():
    assert cli.pair_inputs(["a", "b"], ["x", "y"]) == [("a", "x"), ("b", "y")]


def test_pair_inputs_broadcasts_single_style():
    assert cli.pair_inputs(["a", "b", "c"], ["s"]) == [("a", "s"), ("b", "s"), ("c", "s")]


def test_pair_inputs_mismatched_counts_error():
    with pytest.raises(SystemExit):
        cli.pair_inputs(["a", "b"], ["x", "y", "z"])


def _write_image(path):
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def fake_backbone(monkeypatch):
    # Stand in for a real backbone, targeting layers TinyExtractor exposes.
    monkeypatch.setitem(cli.DEFAULTS, "vgg19", (["b"], ["a", "b"]))
    monkeypatch.setattr(cli, "build_extractor", lambda *a, **k: TinyExtractor())


def _base_args(extra):
    return [
        "--backbone",
        "vgg19",
        "--image-size",
        "16",
        "--optimizer",
        "adam",
        "--lr",
        "0.1",
        "--steps",
        "2",
        "--device",
        "cpu",
        *extra,
    ]


def test_main_single_pair_writes_output(tmp_path, fake_backbone):
    content, style, out = tmp_path / "c.png", tmp_path / "s.png", tmp_path / "result.jpg"
    _write_image(content)
    _write_image(style)

    cli.main(_base_args(["--content", str(content), "--style", str(style), "--output", str(out)]))

    assert out.exists() and Image.open(out).size == (16, 16)


def test_main_batch_writes_one_file_per_pair(tmp_path, fake_backbone):
    contents = [tmp_path / "a.png", tmp_path / "b.png"]
    style = tmp_path / "s.png"
    for p in [*contents, style]:
        _write_image(p)
    out_dir = tmp_path / "out"

    cli.main(
        _base_args(
            ["--content", *map(str, contents), "--style", str(style), "--out-dir", str(out_dir)]
        )
    )

    # One style broadcast over two contents -> two named output files.
    assert (out_dir / "a__s.jpg").exists()
    assert (out_dir / "b__s.jpg").exists()
