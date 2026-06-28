"""Public-API contract tests — catch accidental breaking changes.

Behavioural tests check what the code *does*; these check that its public
*surface* stays stable. Each snapshot below is the committed contract: a
rename, a reordered field, a changed default, or a dropped ``__all__`` export
makes the matching assertion fail, forcing the change to be deliberate (and
this file to be updated alongside it).
"""

from __future__ import annotations

import inspect
import dataclasses

from collections.abc import Callable

import style_transfer as st

# Sentinels standing in for a parameter / field that carries no default value.
REQUIRED = "<required>"
FACTORY = "<factory>"

########################################
#               Helpers                #
########################################


def _field_default(field: dataclasses.Field[object]) -> object:
    # ── A dataclass field is required, factory-defaulted, or literal ──
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return FACTORY
    return REQUIRED


def _signature_contract(
    fn: Callable[..., object],
) -> list[tuple[str, str, object]]:
    # ── (name, kind, default) per parameter — kind pins the `*` boundary ──
    contract: list[tuple[str, str, object]] = []
    for p in inspect.signature(fn).parameters.values():
        empty = p.default is inspect.Parameter.empty
        contract.append((p.name, p.kind.name, REQUIRED if empty else p.default))
    return contract


########################################
#         Public API contract          #
########################################


def test_all_names_are_present() -> None:
    # ── Every name `__all__` promises must actually exist on the package ──
    for name in st.__all__:
        assert hasattr(st, name), f"__all__ promises {name!r} but it is missing"


def test_transferconfig_contract() -> None:
    # ── Field order + defaults are the public config contract ──
    fields = dataclasses.fields(st.TransferConfig)
    contract = [(f.name, _field_default(f)) for f in fields]
    assert contract == [
        ("content_layers", REQUIRED),
        ("style_layers", REQUIRED),
        ("content_weights", FACTORY),
        ("style_weights", FACTORY),
        ("content_weight", 1.0),
        ("style_weight", 1e6),
        ("tv_weight", 1.0),
        ("optimizer", "lbfgs"),
        ("learning_rate", 1.0),
        ("steps", 50),
    ]


def test_build_extractor_contract() -> None:
    # ── Backbone is required; weights default to the pretrained set ──
    assert _signature_contract(st.build_extractor) == [
        ("backbone", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("weights", "POSITIONAL_OR_KEYWORD", "DEFAULT"),
    ]


def test_run_style_transfer_contract() -> None:
    # ── Four positional inputs, then a keyword-only optional callback ──
    assert _signature_contract(st.run_style_transfer) == [
        ("content", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("style", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("extractor", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("cfg", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("callback", "KEYWORD_ONLY", None),
    ]
