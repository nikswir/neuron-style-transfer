"""Tests for the mutant-report helper (flagged_mutants)."""

from __future__ import annotations

import ast
import pytest
import flagged_mutants

from pathlib import Path


def test_func_source_extracts_named_function() -> None:
    src = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    tree = ast.parse(src)
    assert (
        flagged_mutants._func_source(src, tree, "b") == "def b():\n    return 2"
    )


def test_flagged_mutants_parses_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = (
        "    pkg.mod.x_a__mutmut_1: survived\n"
        "    pkg.mod.x_a__mutmut_2: no tests\n"  # dropped (pytest owns this)
        "    pkg.mod.x_a__mutmut_3: timeout\n"
        "    pkg.mod.x_a__mutmut_4: suspicious\n"
        "    1.2 mutants/s: done\n"  # has ': ' but is not a mutant id
        "    not a mutant line\n"
    )

    class _Proc:
        stdout = fake

    monkeypatch.setattr(
        flagged_mutants.subprocess,
        "run",
        lambda *a, **k: _Proc(),
    )
    assert flagged_mutants.flagged_mutants() == [
        ("pkg.mod.x_a__mutmut_1", "survived"),
        ("pkg.mod.x_a__mutmut_3", "timeout"),
        ("pkg.mod.x_a__mutmut_4", "suspicious"),
    ]


def test_mutant_diff_shows_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "mutants" / "src" / "m"
    pkg.mkdir(parents=True)
    (pkg / "f.py").write_text(
        "def x_g__mutmut_orig():\n    return 1\n\n\n"
        "def x_g__mutmut_1():\n    return 2\n",
    )
    monkeypatch.setattr(flagged_mutants, "MUTANTS", tmp_path / "mutants")

    diff = flagged_mutants.mutant_diff("src.m.f.x_g__mutmut_1")
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_func_source_returns_empty_when_absent() -> None:
    src = "def a():\n    return 1\n"
    tree = ast.parse(src)
    assert flagged_mutants._func_source(src, tree, "missing") == ""


def test_mutant_diff_unrecognized_id() -> None:
    # An id whose function name lacks the __mutmut_ marker has no base.
    assert flagged_mutants.mutant_diff("pkg.mod.plain") == "(unrecognized id)"


def test_mutant_diff_source_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The id parses but no matching source exists under MUTANTS/.
    monkeypatch.setattr(flagged_mutants, "MUTANTS", tmp_path / "empty")
    diff = flagged_mutants.mutant_diff("src.m.f.x_g__mutmut_1")
    assert "source not found" in diff


def test_main_reports_no_flagged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(flagged_mutants, "flagged_mutants", lambda: [])
    assert flagged_mutants.main([]) == 0
    assert "no flagged mutants" in capsys.readouterr().out


def test_main_dumps_flagged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        flagged_mutants,
        "flagged_mutants",
        lambda: [("pkg.mod.x_a__mutmut_1", "survived")],
    )
    monkeypatch.setattr(
        flagged_mutants,
        "mutant_diff",
        lambda mutant_id: "-old\n+new",
    )
    assert flagged_mutants.main([]) == 0

    out = capsys.readouterr().out
    assert "1 flagged mutants" in out
    assert "survived" in out
    assert "pkg.mod.x_a__mutmut_1" in out
