"""Tests for the import-hygiene check (check_imports, code-style §6)."""

from __future__ import annotations

import check_imports

from pathlib import Path

FIRST_PARTY = {"style_transfer"}


def _check(tmp_path: Path, text: str) -> bool:
    path = tmp_path / "sample.py"
    path.write_text(text)
    return check_imports.check_file(path, FIRST_PARTY)


def test_relative_import_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "from .foo import bar\n") is True


def test_clean_ladder_passes(tmp_path: Path) -> None:
    assert _check(tmp_path, "import ast\nimport argparse\n") is False


def test_line_ladder_break_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "import argparse\nimport ast\n") is True


def test_block_order_flagged(tmp_path: Path) -> None:
    assert (
        _check(tmp_path, "from style_transfer import x\nimport ast\n") is True
    )


def test_name_ladder_break_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "from x import bbbb, a\n") is True


def test_comment_between_imports_does_not_split_block(tmp_path: Path) -> None:
    # Only a *blank* line separates blocks (§6). A comment between two imports
    # keeps them in one block, so a length-descending pair across the comment
    # must still be flagged -- the old line-gap heuristic treated the comment as
    # a separator and hid the ladder break.
    assert _check(tmp_path, "import argparse\n# a note\nimport ast\n") is True


def test_blank_line_still_separates_blocks(tmp_path: Path) -> None:
    # A genuine blank line *does* separate blocks, so the same two imports in a
    # descending order are no longer compared and the block is clean.
    assert _check(tmp_path, "import argparse\n\nimport ast\n") is False


def test_bare_relative_import_flagged(tmp_path: Path) -> None:
    # `from . import x` has module=None (root ""); still a relative import.
    assert _check(tmp_path, "from . import x\n") is True


def test_future_import_orders_first(tmp_path: Path) -> None:
    # The __future__ block (category 0) precedes a third-party import.
    text = "from __future__ import annotations\n\nimport ast\n"
    assert _check(tmp_path, text) is False


def test_non_import_statements_are_skipped(tmp_path: Path) -> None:
    # A non-import node between imports is ignored, not treated as an import.
    assert _check(tmp_path, "import ast\nx = 1\nimport argparse\n") is False


def test_syntax_error_is_ignored(tmp_path: Path) -> None:
    # A file ruff/python can't parse is left to them, not flagged here.
    assert _check(tmp_path, "import (\n") is False


def test_multiline_import_needs_blank_line(tmp_path: Path) -> None:
    # A parenthesised multi-line import sharing a block with another import
    # must be set off by a blank line -- here it is not, so it is flagged.
    text = "from x import a\nfrom y import (\n    b,\n    cc,\n)\n"
    assert _check(tmp_path, text) is True


def test_no_imports_passes(tmp_path: Path) -> None:
    # A file with no imports yields no blocks (exercises the empty-blocks path).
    assert _check(tmp_path, "x = 1\ny = 2\n") is False


def test_first_party_roots_discovers_local_packages(
    tmp_path: Path,
) -> None:
    # Detected from disk, not hard-coded: a top-level `<pkg>/__init__.py`
    # package and a sibling module both count as first-party. No `src/` here,
    # so the missing-directory branch is exercised too.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (tmp_path / "solo.py").write_text("")

    roots = check_imports.first_party_roots(tmp_path)
    assert "mypkg" in roots
    assert "solo" in roots


def test_main_flags_bad_and_passes_clean(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    good = tmp_path / "good.py"
    bad.write_text("from .foo import bar\n")
    good.write_text("import ast\nimport argparse\n")
    assert check_imports.main([str(bad)]) == 1
    assert check_imports.main([str(good)]) == 0
