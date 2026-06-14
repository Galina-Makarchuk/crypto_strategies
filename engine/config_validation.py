"""Shared validation rules for the config dataclasses.

The per-strategy ``*Params`` classes and ``TradingConfig`` keep their own
``__post_init__`` *hook* (that hook is what makes both construction **and**
``dataclasses.replace`` re-validate, which is the load-bearing guarantee for
notebook overrides and parameter sweeps) and delegate the actual *rules* here,
so they read identically across configs and live in one place. (``DataSpec``
keeps its own bespoke interval/category enum checks.)

The policy mirrors what the configs already enforce: reject only genuinely
invalid values (negatives, zero where a value must be positive, bad
categoricals, out-of-range indices, empty required strings) — never an upper cap
or a cross-field ordering, so sweeps stay free to roam. ``bool`` is excluded from
the int/number checks because ``bool`` subclasses ``int``.
"""

from __future__ import annotations

from typing import Iterable


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def require(owner: str, name: str, value, ok: bool, requirement: str) -> None:
    """Raise a uniform ValueError unless ``ok``. The single funnel every helper
    below goes through, so error messages are identical across all configs."""
    if not ok:
        raise ValueError(
            f"{owner}.{name}={value!r} is invalid: must be {requirement}"
        )


def positive_int(owner: str, name: str, value) -> None:
    require(owner, name, value, _is_int(value) and value >= 1,
            "a positive int (>= 1)")


def non_negative_int(owner: str, name: str, value) -> None:
    require(owner, name, value, _is_int(value) and value >= 0,
            "a non-negative int (>= 0)")


def optional_positive_int(owner: str, name: str, value) -> None:
    require(owner, name, value, value is None or (_is_int(value) and value >= 1),
            "None or a positive int (>= 1)")


def optional_positive_number(owner: str, name: str, value) -> None:
    require(owner, name, value, value is None or (_is_num(value) and value > 0),
            "None or a positive number (> 0)")


def optional_non_negative_number(owner: str, name: str, value) -> None:
    require(owner, name, value, value is None or (_is_num(value) and value >= 0),
            "None or a non-negative number (>= 0)")


def positive_number(owner: str, name: str, value) -> None:
    require(owner, name, value, _is_num(value) and value > 0,
            "a positive number (> 0)")


def non_negative_number(owner: str, name: str, value) -> None:
    require(owner, name, value, _is_num(value) and value >= 0,
            "a non-negative number (>= 0)")


def in_range(owner: str, name: str, value, lo: float, hi: float) -> None:
    require(owner, name, value, _is_num(value) and lo <= value <= hi,
            f"in [{lo}, {hi}]")


def one_of(owner: str, name: str, value, choices: Iterable) -> None:
    choices = tuple(choices)
    require(owner, name, value, value in choices,
            "one of " + " | ".join(repr(c) for c in choices))


def enum_member(owner: str, name: str, value, enum_cls) -> None:
    """Require an actual member of ``enum_cls`` — not its raw ``.value`` string.
    Passing e.g. ``direction='long'`` instead of ``TradeDirection.LONG`` is a
    natural mistake that would otherwise sail through (the string is in no
    direction tuple, silently gating out every side); this makes it fail loudly
    at construction like every other bad knob."""
    require(owner, name, value, isinstance(value, enum_cls),
            f"a {enum_cls.__name__} member (" +
            " | ".join(f"{enum_cls.__name__}.{m.name}" for m in enum_cls) +
            f"), not {type(value).__name__}")


def non_empty_str(owner: str, name: str, value) -> None:
    require(owner, name, value, isinstance(value, str) and bool(value),
            "a non-empty string")
