"""Tests for the device-agnostic check (check_device)."""

from __future__ import annotations

import check_device

from pathlib import Path


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(text)
    return path


def test_flags_cuda_call(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.cuda()\n")
    assert check_device.check_file(path) is True


def test_flags_to_cuda_string(tmp_path: Path) -> None:
    path = _write(tmp_path, 'x = t.to("cuda")\n')
    assert check_device.check_file(path) is True


def test_flags_device_kwarg(tmp_path: Path) -> None:
    path = _write(tmp_path, 'f(device="cuda")\n')
    assert check_device.check_file(path) is True


def test_flags_indexed_cuda(tmp_path: Path) -> None:
    # An explicit device index (cuda:0, cuda:1) is still a hard-code: the old
    # patterns required a closing quote right after `cuda`, so `cuda:0` slipped
    # through the gate built to catch exactly this.
    assert check_device.check_file(_write(tmp_path, 'x = t.to("cuda:0")\n'))
    assert check_device.check_file(_write(tmp_path, 'f(device="cuda:1")\n'))


def test_flags_to_torch_device_cuda(tmp_path: Path) -> None:
    # The most idiomatic hard-code -- pinning a tensor onto a freshly built
    # device object -- must be flagged too.
    path = _write(tmp_path, 'x = t.to(torch.device("cuda"))\n')
    assert check_device.check_file(path) is True
    path = _write(tmp_path, 'x = t.to(torch.device("cuda:0"))\n')
    assert check_device.check_file(path) is True


def test_bare_torch_device_selection_not_flagged(tmp_path: Path) -> None:
    # Device *selection* via torch.device(...) is intentionally exempt (that is
    # how `pick_device` resolves the accelerator); only pinning it onto a tensor
    # with `.to(...)` is a smell.
    path = _write(tmp_path, 'dev = torch.device("cuda")\n')
    assert check_device.check_file(path) is False


def test_device_ok_exempts(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.cuda()  # device-ok\n")
    assert check_device.check_file(path) is False


def test_clean_file_passes(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.to(dev)\n")
    assert check_device.check_file(path) is False


def test_main_flags_bad_and_passes_clean(tmp_path: Path) -> None:
    bad = _write(tmp_path, "x = t.cuda()\n")
    good = tmp_path / "good.py"
    good.write_text("x = t.to(dev)\n")
    assert check_device.main([str(bad)]) == 1
    assert check_device.main([str(good)]) == 0
