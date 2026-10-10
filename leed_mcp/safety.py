# leed_mcp/safety.py
"""Output safety (spec 9): text taken from project files or logs is data.

Tools spec, Security: *"Servers MUST ... Sanitize tool outputs."* Every
string in a result passes through ``sanitize`` in ``envelope.emit``: control
characters other than newline and tab are removed, and one string holds at
most ``FIELD_MAX`` characters, with the cut marked.
"""
from __future__ import annotations

import re
from typing import Any, Final

__all__ = ["FIELD_MAX", "clean_text", "sanitize"]

#: At most this many characters in one string field. Chosen (spec 9).
FIELD_MAX: Final = 2000
#: Unicode category Cc (C0, DEL, C1) except tab (\x09) and newline (\x0a).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_text(text: str, limit: int = FIELD_MAX) -> str:
    """``text`` without control characters (newline and tab kept), cut to
    ``limit`` characters; a cut ends with ``[cut: N more characters]``."""
    text = _CONTROL.sub("", text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]} [cut: {len(text) - limit} more characters]"


def sanitize(value: Any) -> Any:
    """``clean_text`` on every string in a JSON-shaped value, keys included."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return _sanitize_dict(value)
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def _sanitize_dict(value: dict) -> dict:
    """``sanitize`` on each key and value. Two keys that become the same
    after cleaning both stay: a later one gets `` [2]``, `` [3]``, ..."""
    out: dict = {}
    for key, item in value.items():
        if isinstance(key, str):
            base, n = clean_text(key), 1
            key = base
            while key in out:
                n += 1
                key = f"{base} [{n}]"
        out[key] = sanitize(item)
    return out
