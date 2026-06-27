"""Tests for the structural-comment check (check_banners, code-style §1-§2)."""

from __future__ import annotations

import check_banners

from pathlib import Path

PATH = Path("sample.py")


def test_consistent_banner_passes() -> None:
    lines = [
        "#" * 30,
        "#" + "Section".center(28) + "#",
        "#" * 30,
    ]
    assert check_banners.check_banners(PATH, lines) is False


def test_mixed_widths_flagged() -> None:
    lines = [
        "#" * 30,
        "#" + "A".center(28) + "#",
        "#" * 30,
        "",
        "#" * 20,
        "#" + "B".center(18) + "#",
        "#" * 20,
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_intro_detached_below_flagged() -> None:
    lines = ["# ── Step ──────", "", "x = 1"]
    assert check_banners.check_intros(PATH, lines) is True


def test_intro_well_formed_passes() -> None:
    lines = ["def f():", "    # ── Step ──────", "    x = 1"]
    assert check_banners.check_intros(PATH, lines) is False


def test_middle_width_mismatch_flagged() -> None:
    # Borders are width 30 but the middle line is narrower -> flagged.
    lines = [
        "#" * 30,
        "#" + "A".center(20) + "#",
        "#" * 30,
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_label_not_centred_flagged() -> None:
    # Width matches but the label hugs the left edge (pad 0 / 23) -> flagged.
    lines = [
        "#" * 30,
        "#" + "Label" + " " * 23 + "#",
        "#" * 30,
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_banner_indent_mismatch_flagged() -> None:
    # Banner indented 4, but the code it heads sits at column 0 -> flagged.
    lines = [
        "    " + "#" * 30,
        "    #" + "A".center(28) + "#",
        "    " + "#" * 30,
        "x = 1",
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_intro_needs_blank_line_above_flagged() -> None:
    # An intro directly under a plain statement (no blank, not a suite header).
    lines = ["x = 1", "# ── Step ──────", "y = 2"]
    assert check_banners.check_intros(PATH, lines) is True


def test_main_flags_bad_and_passes_clean(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    good = tmp_path / "good.py"
    bad.write_text(
        "\n".join(
            [
                "#" * 30,
                "#" + "A".center(20) + "#",
                "#" * 30,
            ],
        )
        + "\n",
    )
    good.write_text("x = 1\n")
    assert check_banners.main([str(bad)]) == 1
    assert check_banners.main([str(good)]) == 0
