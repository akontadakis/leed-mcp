# leed_daylight/build/_names.py
"""The one shared name-slugging rule for layer 3 (build).

Phase 4 builds file names and shell arguments from a room's or a window
group's own name, so the rule that turns ``"Room A"`` into ``"Room_A"``
must be exactly the same everywhere it is applied. Before this module
existed, ``build.groups`` and ``build.prepare`` each defined their own
copy of the same regular expression; two copies can only ever drift apart,
never merge back (finding I4).
"""
from __future__ import annotations

import re

#: A generated name must be filesystem- and shell-safe.
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def slug(name: str) -> str:
    """Replace every character outside ``[A-Za-z0-9_-]`` with ``"_"``."""
    return _SLUG_SAFE.sub("_", name)
