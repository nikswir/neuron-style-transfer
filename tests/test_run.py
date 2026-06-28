"""Tests for the Hydra entry point's core run.

`run()` is exercised with a tiny stand-in extractor so the wiring -- composed
config -> TransferConfig -> optimization -> writing the output -- runs
end-to-end without downloading pretrained weights.
"""

from __future__ import annotations

import torch
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


def test_run_propagates_picked_device(tmp_path, monkeypatch):
    # The resolved device must reach build_extractor(...).to(device) and every
    # load_image(...) call. On CPU a dropped device (.to(None), load_image(...,
    # None)) is a silent no-op, so the size-only smoke test cannot catch it;
    # spy on the device handed to each seam explicitly.
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_image(content)
    _write_image(style)

    sentinel = torch.device("cpu")
    monkeypatch.setattr(run_module, "pick_device", lambda requested: sentinel)

    seen: dict = {}

    class _SpyExtractor(TinyExtractor):
        def to(self, device):
            seen["extractor_to"] = device
            return self

    monkeypatch.setattr(
        run_module,
        "build_extractor",
        lambda *a, **k: _SpyExtractor(),
    )

    real_load = run_module.load_image

    def _spy_load(path, image_size, device):
        seen.setdefault("load_devices", []).append(device)
        return real_load(path, image_size, device)

    monkeypatch.setattr(run_module, "load_image", _spy_load)
    monkeypatch.setattr(
        run_module,
        "run_style_transfer",
        lambda content_t, style_t, extractor, cfg: TransferResult(
            image=content_t,
            history={},
        ),
    )

    run_module.run(_cfg(content, style), tmp_path / "run")

    assert seen["extractor_to"] is sentinel
    assert seen["load_devices"] == [sentinel, sentinel]


def test_run_creates_nested_output_dir_and_is_idempotent(tmp_path, monkeypatch):
    # out_dir.mkdir(parents=True, exist_ok=True): without `parents` a nested
    # path raises FileNotFoundError, without `exist_ok` a re-run into the
    # existing dir raises FileExistsError. Pin both flags here.
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_image(content)
    _write_image(style)

    monkeypatch.setattr(
        run_module,
        "build_extractor",
        lambda *a, **k: TinyExtractor(),
    )

    out_dir = tmp_path / "deep" / "nested" / "run"  # parents do not exist yet
    first = run_module.run(_cfg(content, style), out_dir)
    assert first.exists()

    # Re-running into the now-existing nested directory must not raise.
    second = run_module.run(_cfg(content, style), out_dir)
    assert second.exists()


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
