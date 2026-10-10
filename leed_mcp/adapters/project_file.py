# leed_mcp/adapters/project_file.py
"""``inputs(action="update", file="project")`` (spec 6.3).

The change is ``values`` (dotted keys to JSON values) and ``unset`` (keys
to remove), each key from ``project_keys``. The file is written by the
pipeline's ``render``, which lays it out as the template and quotes every
string, so clock times and dates stay strings. Comments the template does
not have are lost by a write, so the update is refused while there are any
unless ``drop_hand_comments``. The ``.part`` file is checked with
``check_project_draft`` (``FILL_ME`` allowed). The ``source`` sentence is
appended to ``_run/input_log.jsonl`` with the keys and the time.
"""
from __future__ import annotations

import copy
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from leed_daylight.edit.project_yaml import hand_comments, render, set_path
from leed_daylight.project.config import check_project_draft
from leed_mcp.adapters.input_files import check_project_paths, log_source, write_file
from leed_mcp.adapters.input_read import project_content
from leed_mcp.adapters.project_keys import check_keys
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused

__all__ = ["SOURCE_MAX", "update_project"]

#: At most this many characters in one ``source`` sentence. Chosen.
SOURCE_MAX = 1000
#: How many of the file's own comments the refusal names.
_COMMENTS_SHOWN = 3


def _remove(raw: dict, path: str) -> None:
    parts = path.split(".")
    target: Any = raw
    for part in parts[:-1]:
        target = target.get(part) if isinstance(target, dict) else None
    if isinstance(target, dict):
        target.pop(parts[-1], None)


def _check_source(source: str) -> str:
    source = source.strip()
    if not source:
        raise Refused(refusal("INPUT_INVALID", "source is blank; say in one sentence where the values came "
                              "from, for example 'typed by the user in chat'.", field="source"))
    if len(source) > SOURCE_MAX:
        raise Refused(refusal("INPUT_INVALID", f"source holds {len(source)} characters; one sentence of at "
                              f"most {SOURCE_MAX} is enough.", field="source"))
    return source


def update_project(folder: pathlib.Path, expected: str | None, values: Mapping[str, Any],
                   unset: Sequence[str], source: str, drop_hand_comments: bool) -> dict[str, Any]:
    """Set ``values`` and remove ``unset`` in project.yaml; returns
    ``{"written", "content", "fill_me", "log"}``."""
    check_keys(values, unset)
    source = _check_source(source)
    held: dict[str, Any] = {}

    def build(old: bytes | None, raw: Mapping[str, Any] | None) -> bytes:
        if old is None:
            raise Refused(refusal("FILE_MISSING", f"{folder}: no project.yaml; run project(action='create') "
                                  "first.", field="file"))
        hand = hand_comments(old.decode("utf-8"))
        if hand and not drop_hand_comments:
            shown = "; ".join(hand[:_COMMENTS_SHOWN]) + ("; ..." if len(hand) > _COMMENTS_SHOWN else "")
            raise Refused(refusal(
                "HAND_COMMENTS_WOULD_BE_LOST",
                f"project.yaml holds {len(hand)} comment(s) the template does not have ({shown}); a write lays "
                "the file out as the template and loses them. Ask the user, then send drop_hand_comments: "
                "true to write anyway.", field="drop_hand_comments"))
        new = copy.deepcopy(dict(raw))
        for key, value in values.items():
            set_path(new, key, copy.deepcopy(value))
        for key in unset:
            _remove(new, key)
        check_project_paths(folder, new)
        data = render(new).encode("utf-8")
        held["data"] = data
        return data

    def check(part: pathlib.Path) -> None:
        errors, unfinished = check_project_draft(part)
        if errors:
            raise Refused(refusal("INPUT_INVALID", f"the new project.yaml is refused: {'; '.join(errors)}",
                                  field="values"))
        held["fill_me"] = list(unfinished)

    written = write_file(folder, "project.yaml", expected, build, check)
    log = log_source(folder, name="project.yaml", keys=list(values), unset=list(unset), source=source)
    return {"written": written, "content": project_content(held["data"]).content,
            "fill_me": held["fill_me"], "log": log}
