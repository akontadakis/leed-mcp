# leed_daylight/drivers/_folder.py
"""Shared folder checks for driver commands (layer 7): private to
`drivers/` because "a project folder must not sit inside the repository"
is one rule, defined once, that every driver command enforces the same
way -- `run.py` first, `rooms.py` alongside it.
"""
from __future__ import annotations

import pathlib
import sys
import tomllib

from ..engine.jobs import checked_dir

#: The folders under a prepared model's output folder whose stale files the
#: model writer deletes (``build.prepare``: ``*.rad`` and ``*.pts``).
PREPARED_DIRS = (("blinds",), ("grids",))

def _is_repository(folder: pathlib.Path) -> bool:
    try:
        data = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("project", {}).get("name") == "leed-mcp"


def repository_root() -> pathlib.Path:
    """This repository's folder. In an editable install it holds this
    package: `leed_daylight/drivers/_folder.py` -> `drivers` ->
    `leed_daylight` -> the repository root. A `--no-editable` install puts
    the package in `.venv/lib/.../site-packages`, so the folder holding the
    running `.venv` is tried next. Neither found: the package's own parent
    folder, the old rule."""
    package_parent = pathlib.Path(__file__).resolve().parents[2]
    for candidate in (package_parent, pathlib.Path(sys.prefix).resolve().parent):
        if _is_repository(candidate):
            return candidate
    return package_parent


#: This repository's folder (``repository_root``).
REPO_ROOT = repository_root()


def check_folder(
    folder: pathlib.Path, repo_root: pathlib.Path, error: type[Exception],
) -> None:
    """Refuse `folder` (raising `error`) if it does not exist, is not a
    folder, or sits inside `repo_root` -- client models must never be
    placed inside the repository.

    `repo_root` is passed in by the caller rather than read from this
    module's own `REPO_ROOT`, so a caller's test can monkeypatch its own
    module-level `REPO_ROOT` (the name it imported from here) and have
    this check see the patched value, exactly as if the check were still
    written out inline in that caller's own module.
    """
    if not folder.is_dir():
        raise error(f"{folder}: not a folder (or does not exist)")
    check_outside_repo(folder, repo_root, error)


def check_outside_repo(
    folder: pathlib.Path, repo_root: pathlib.Path, error: type[Exception],
) -> None:
    """Refuse `folder` (raising `error`) if it is, or sits inside,
    `repo_root`. The folder need not exist yet: `init` checks a folder
    before it creates it."""
    resolved = folder.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        raise error(
            f"{folder}: this project folder is inside the repository "
            f"({repo_root}); client models must never be placed inside the "
            "repository -- put the project folder somewhere else"
        )


def check_output_dirs(folder: pathlib.Path, *chains: tuple[str, ...]) -> None:
    """Refuse (``UnsafePath``), by name, a symbolic link or a non-folder at
    any existing folder of each chain under ``folder`` (``("_run",
    "blinds")`` is ``_run`` and then ``_run/blinds``), before the command
    deletes, renames or replaces anything there: a link would send that
    outside the project folder (re-review OOS-1). Nothing is created."""
    for parts in chains:
        checked_dir(folder, *parts, make=False)
