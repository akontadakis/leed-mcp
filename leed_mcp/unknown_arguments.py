# leed_mcp/unknown_arguments.py
"""Refuse an argument a tool does not name, and a misspelt key inside a dict
argument. Copied from rad_mcp (``src/rad_mcp/unknown_arguments.py``) and
trimmed: the code is ``UNKNOWN_ARGUMENT``; no retired-argument table.

The SDK's argument model ignores a key the tool's signature does not name,
so a misspelt optional argument would run with its default. Pure: no SDK
server import.
"""
from __future__ import annotations

import difflib
from collections.abc import Collection, Mapping
from typing import Any, Final

from mcp_types import CallToolResult

from leed_mcp.envelope import Advisory, refuse

#: `difflib.get_close_matches`' own default. Chosen, not measured.
CLOSE_MATCH_CUTOFF: Final = 0.6
#: At most this many closest names per unknown key. Chosen.
MAX_SUGGESTIONS: Final = 2


def _closest_sentence(key: str, names: list[str]) -> str | None:
    close = difflib.get_close_matches(key, names, n=MAX_SUGGESTIONS, cutoff=CLOSE_MATCH_CUTOFF)
    if len(close) == 1:
        return f"The closest name is {close[0]!r}."
    if close:
        return "The closest names are " + ", ".join(repr(c) for c in close) + "."
    return None


def unknown_argument_message(tool: str, key: str, known: Collection[str]) -> str:
    names = sorted(known)
    parts = [f"{tool} has no argument {key!r}."]
    if (close := _closest_sentence(key, names)) is not None:
        parts.append(close)
    parts.append(f"Its arguments are: {', '.join(names)}." if names else "It takes no arguments.")
    return " ".join(parts)


def unknown_key_message(where: str, key: str, known: Collection[str]) -> str:
    names = sorted(known)
    parts = [f"{where} has no key {key!r}."]
    if (close := _closest_sentence(key, names)) is not None:
        parts.append(close)
    parts.append(f"Its keys are: {', '.join(names)}.")
    return " ".join(parts)


def unknown_key_advisories(where: str, mapping: Mapping[str, Any], known: Collection[str], *,
                           field: str) -> list[Advisory]:
    """One `UNKNOWN_ARGUMENT` advisory per key of `mapping` that `known` does
    not name, sorted; `field` is the top-level argument the dict came in."""
    return [
        Advisory(code="UNKNOWN_ARGUMENT", severity="error", field=field,
                 message=unknown_key_message(where, key, known))
        for key in sorted(str(k) for k in set(mapping) - set(known))
    ]


def refuse_unknown_arguments(tool: str, arguments: Mapping[str, Any],
                             known: Collection[str]) -> CallToolResult | None:
    """The refusal for every key in `arguments` that `known` does not name, or None."""
    unknown = sorted(str(key) for key in set(arguments) - set(known))
    if not unknown:
        return None
    advisories = [
        Advisory(code="UNKNOWN_ARGUMENT", severity="error", field=key,
                 message=unknown_argument_message(tool, key, known))
        for key in unknown
    ]
    return refuse(" ".join(a.message for a in advisories), advisories)
