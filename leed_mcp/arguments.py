# leed_mcp/arguments.py
"""Per-action argument rules and the checks every tool body starts with
(spec 6): an argument the action does not use is ``UNKNOWN_ARGUMENT``; a
required one that is absent is ``ARGUMENT_MISSING``; ``folder`` must lie
inside a root; ``case`` must be a safe case name."""
from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from leed_daylight.project.spaces import CASE_NAME_RULE, is_safe_case_name
from leed_mcp.envelope import Advisory, refuse, refusal, refuse_missing
from leed_mcp.errors import Refused
from leed_mcp.paths import PathOutsideRoots, RootsNotConfigured, contain

__all__ = ["Action", "check_action", "check_case", "resolve_folder"]


@dataclass(frozen=True)
class Action:
    """What one action of a tool takes."""

    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()


def check_action(tool: str, action: str, given: Mapping[str, Any],
                 table: Mapping[str, Action]) -> None:
    """Raise ``Refused`` for an argument ``action`` does not take (a value
    that is not None counts as given) or a required one that is None."""
    rule = table[action]
    call = f"{tool}(action={action!r})"
    takes = sorted(rule.required | rule.optional)
    extra = sorted(k for k, v in given.items() if v is not None and k != "action"
                   and k not in rule.required and k not in rule.optional)
    if extra:
        names = ", ".join(takes) if takes else "no other argument"
        advisories = [
            Advisory(code="UNKNOWN_ARGUMENT", severity="error", field=key,
                     message=f"{call} does not take {key!r}; it takes: {names}.")
            for key in extra
        ]
        raise Refused(refuse(" ".join(a.message for a in advisories), advisories))
    missing = refuse_missing(call, **{key: given.get(key) for key in sorted(rule.required)})
    if missing is not None:
        raise Refused(missing)


def resolve_folder(folder: str, roots: Sequence[pathlib.Path], *, must_exist: bool = True) -> pathlib.Path:
    """``folder`` resolved inside a root; with ``must_exist``, an existing
    folder. Raises ``Refused`` (``ROOTS_NOT_CONFIGURED``, ``PATH_OUTSIDE_ROOTS``,
    ``FILE_MISSING``), each with ``field="folder"``."""
    try:
        path = contain(folder, roots)
    except RootsNotConfigured as exc:
        raise Refused(refusal("ROOTS_NOT_CONFIGURED", str(exc), field="folder")) from None
    except PathOutsideRoots as exc:
        raise Refused(refusal("PATH_OUTSIDE_ROOTS", str(exc), field="folder")) from None
    if must_exist and not path.is_dir():
        raise Refused(refusal("FILE_MISSING", f"{path}: no such project folder.", field="folder"))
    return path


def check_case(case: str, field: str = "case") -> str:
    """``case`` when it is a safe case name, else ``Refused`` (``INPUT_INVALID``)."""
    if not is_safe_case_name(case):
        raise Refused(refusal("INPUT_INVALID", f"{field} {case!r}: {CASE_NAME_RULE}", field=field))
    return case
