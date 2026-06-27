"""Tests for the Hydra entry point's core run.

`run()` is exercised with a tiny stand-in extractor so the wiring -- composed
config -> TransferConfig -> optimization -> writing the output -- runs
end-to-end without downloading pretrained weights.
"""

from __future__ import annotations

import numpy as np

from PIL import Image
from omegaconf import OmegaConf

from style_transfer import run as run_module
from tests.test_transfer import TinyExtractor
from style_transfer.transfer import TransferResult

########################################
#               Helpers                #
########################################


def _write_image(path):
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(path)


def _cfg(content, style):
    return OmegaConf.create(
        {
            "backbone": {
                "name": "tiny",
                "content_layers": ["b"],
                "style_layers": ["a", "b"],
            },
            "data": {"content": str(content), "style": str(style)},
            "transfer": {
                "content_weight": 1.0,
                "style_weight": 1.0,
                "tv_weight": 1.0,
                "optimizer": "adam",
                "learning_rate": 0.1,
                "steps": 2,
            },
            "runtime": {"image_size": 16, "device": "cpu", "output": "out.jpg"},
        },
    )


########################################
#               Core run               #
########################################


def test_run_writes_output_into_out_dir(tmp_path, monkeypatch):
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_image(content)
    _write_image(style)

    # Stand in for the real backbone so no weights are downloaded.
    monkeypatch.setattr(
        run_module,
        "build_extractor",
        lambda *a, **k: TinyExtractor(),
    )

    out_dir = tmp_path / "run"
    out_path = run_module.run(_cfg(content, style), out_dir)

    assert out_path == out_dir / "out.jpg"
    assert Image.open(out_path).size == (16, 16)


def test_run_bridges_config_into_transfer_config(tmp_path, monkeypatch):
    # The refactor's core seam: every backbone/transfer field must flow into
    # the TransferConfig handed to the optimizer. Capture that config and
    # assert each field, so dropping any kwarg (which falls back to a
    # TransferConfig default) is caught -- the size-only smoke test cannot.
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_image(content)
    _write_image(style)

    cfg = _cfg(content, style)
    # Distinct, non-default values so a dropped kwarg changes the captured one.
    cfg.transfer.content_weight = 2.0
    cfg.transfer.tv_weight = 5.0

    captured: dict = {}

    def _capture(content_t, style_t, extractor, transfer_cfg):
        captured["cfg"] = transfer_cfg
        captured["content"] = content_t
        captured["style"] = style_t
        return TransferResult(image=content_t, history={})

    def _build(name, *a, **k):
        captured["backbone"] = name
        return TinyExtractor()

    monkeypatch.setattr(run_module, "build_extractor", _build)
    monkeypatch.setattr(run_module, "run_style_transfer", _capture)

    run_module.run(cfg, tmp_path / "run")

    # The backbone name must reach build_extractor (not a dropped / None arg).
    assert captured["backbone"] == "tiny"

    tc = captured["cfg"]
    assert tc.content_layers == ["b"]
    assert tc.style_layers == ["a", "b"]
    assert tc.content_weight == 2.0
    assert tc.style_weight == 1.0
    assert tc.tv_weight == 5.0
    assert tc.optimizer == "adam"
    assert tc.learning_rate == 0.1
    assert tc.steps == 2

    # load_image must use cfg.runtime.image_size (16); a dropped size -> 512.
    assert captured["content"].shape == (1, 3, 16, 16)
    assert captured["style"].shape == (1, 3, 16, 16)


########################################
#             Entry point              #
########################################


def test_main_writes_into_hydra_output_dir(tmp_path, monkeypatch):
    # Drive the @hydra.main wrapper: passing a config straight to the decorated
    # `main` bypasses CLI composition, so with HydraConfig patched to supply the
    # per-run dir and the backbone stubbed, main() must write the output there.
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_image(content)
    _write_image(style)

    out_dir = tmp_path / "hydra_run"

    class _Runtime:
        output_dir = str(out_dir)

    class _HydraConf:
        runtime = _Runtime()

    monkeypatch.setattr(
        run_module.HydraConfig,
        "get",
        staticmethod(lambda: _HydraConf()),
    )
    monkeypatch.setattr(
        run_module,
        "build_extractor",
        lambda *a, **k: TinyExtractor(),
    )

    run_module.main(_cfg(content, style))

    assert (out_dir / "out.jpg").exists()
