# leed_daylight/edit/files.py
"""How the wizard touches files (wizard spec 3, 7.1, 10).

- The path rule: every path the browser sends is resolved inside the
  project folder at the moment of use (``resolve_inside``); no absolute
  path, no ``..``, no symlink leading out.
- Change detection: the sha256 of each file is remembered when a step
  serves it (``Served``); a save is refused when the file on disk differs,
  so nothing is overwritten silently.
- Safe saving (``save_files``): each new file is written to ``<name>.part``
  in its own folder, the step's check runs on the ``.part`` files, the old
  file is backed up to ``_run/backup/<name>.<YYYYmmdd-HHMMSS>`` (the
  newest 20 kept per file), then ``os.replace`` puts the new one in place.
  Every file it writes is created new and never through a link, and the
  ``_run`` folders it writes in must be real folders inside the project
  (security review H1): a planted link is refused, nothing is written
  outside the folder.
- Recent projects: ``~/.leed-daylight/recent.json``, a JSON list of at most
  10 absolute folder paths and nothing else, readable by its owner only.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import threading
from collections.abc import Callable, Mapping

from ..engine.jobs import checked_dir, replace_from_part, write_new
from ..scene.input_file import read_input_bytes

BACKUP_DIR = ("_run", "backup")
KEEP_BACKUPS = 20
PART_SUFFIX = ".part"
RECENT_MAX = 10
WIZARD_LOG = ("_run", "wizard.log")

#: A backup's name after the file's own name: the stamp, then -2, -3 on a clash.
_BACKUP_TAIL = re.compile(r"\.(\d{8}-\d{6})(?:-(\d+))?$")
#: One save at a time: two browser tabs saving together must not interleave.
_SAVE_LOCK = threading.Lock()


class PathOutside(ValueError):
    """A path from the browser that does not stay inside the project folder."""


class ChangedOutside(ValueError):
    """A file changed on disk since the wizard served it."""


def resolve_inside(folder: pathlib.Path, rel: str) -> pathlib.Path:
    """The resolved path of ``rel`` inside ``folder``, symlinks followed.
    Refuses an empty, absolute, drive or UNC path, a ``..`` part, and a
    path that resolves outside ``folder``. Callers open the returned path."""
    if not isinstance(rel, str) or not rel.strip():
        raise PathOutside(f"{rel!r}: give a path inside the project folder")
    windows = pathlib.PureWindowsPath(rel)
    if rel.startswith(("/", "\\")) or windows.drive or pathlib.PurePosixPath(rel).is_absolute():
        raise PathOutside(f"{rel}: give a path relative to the project folder, not an absolute one")
    if ".." in re.split(r"[\\/]", rel):
        raise PathOutside(f"{rel}: a path may not use '..'; give a path inside the project folder")
    base = pathlib.Path(folder).resolve()
    resolved = (base / rel).resolve()
    if resolved != base and base not in resolved.parents:
        raise PathOutside(f"{rel}: leads outside the project folder ({base})")
    return resolved


def sha256_or_none(path: pathlib.Path) -> str | None:
    """The sha256 of a file's bytes, or None when there is no such file."""
    try:
        return hashlib.sha256(read_input_bytes(path)).hexdigest()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


class Served:
    """The sha256 of every file as a step served it, per folder and name.
    A file that did not exist is remembered as None, so a first save may
    create it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._digests: dict[tuple[str, str], str | None] = {}

    @staticmethod
    def _key(folder, name: str) -> tuple[str, str]:
        return str(pathlib.Path(folder).resolve()), name

    def remember(self, folder, name: str) -> str | None:
        digest = sha256_or_none(resolve_inside(pathlib.Path(folder), name))
        with self._lock:
            self._digests[self._key(folder, name)] = digest
        return digest

    def check(self, folder, name: str) -> None:
        """Refuse (``ChangedOutside``) a file that differs from what was
        served, or that was never served in this start of the wizard."""
        with self._lock:
            key = self._key(folder, name)
            known = key in self._digests
            served = self._digests.get(key)
        if not known or served != sha256_or_none(resolve_inside(pathlib.Path(folder), name)):
            raise ChangedOutside(f"{name} was changed outside the wizard; reload the step")

    def clear(self) -> None:
        with self._lock:
            self._digests.clear()


#: The one ``Served`` of a running wizard.
SERVED = Served()


def _backup_folder(folder: pathlib.Path, name: str = "", *, make: bool = False) -> pathlib.Path:
    """The backup folder of ``name`` (its sub-folders kept), each part a
    real folder inside the project, never a link; ``make`` creates it."""
    parts = pathlib.PurePosixPath(name.replace("\\", "/")).parent.parts if name else ()
    return checked_dir(folder, *BACKUP_DIR, *parts, make=make)


def _backups_of(stem: pathlib.Path) -> list[tuple[tuple[str, int], pathlib.Path]]:
    """The backups of one file, oldest first, each with its (stamp, number)."""
    if not stem.parent.is_dir():
        return []
    found = []
    for path in stem.parent.iterdir():
        if not path.name.startswith(f"{stem.name}."):
            continue
        match = _BACKUP_TAIL.fullmatch(path.name[len(stem.name):])
        if match and path.is_file():
            found.append(((match.group(1), int(match.group(2) or 1)), path))
    return sorted(found)


def backup(folder, name: str, *, now: datetime.datetime) -> pathlib.Path | None:
    """Copy ``<folder>/<name>`` to ``_run/backup/<name>.<stamp>``
    (``-2``, ``-3`` after the stamp when that second already has one; a
    sub-folder in ``name`` is kept under the backup folder), then keep the
    newest ``KEEP_BACKUPS``. None when there is no file to back up."""
    folder = pathlib.Path(folder)
    source = resolve_inside(folder, name)
    if not source.is_file():
        return None
    stem = _backup_folder(folder, name, make=True) / pathlib.PurePosixPath(name.replace("\\", "/")).name
    stamp = now.strftime("%Y%m%d-%H%M%S")
    taken = [number for (when, number), _ in _backups_of(stem) if when == stamp]
    suffix = f"-{max(taken) + 1}" if taken else ""
    target = stem.with_name(f"{stem.name}.{stamp}{suffix}")
    write_new(target, source.read_bytes(), leftover=False)
    shutil.copystat(source, target)
    prune_backups(folder, name)
    return target


def prune_backups(folder, name: str, keep: int = KEEP_BACKUPS) -> None:
    """Remove all but the newest ``keep`` backups of one file."""
    stem = _backup_folder(folder, name) / pathlib.PurePosixPath(name.replace("\\", "/")).name
    found = _backups_of(stem)
    for _, path in found[:max(len(found) - keep, 0)]:
        path.unlink()


def save_files(folder, contents: Mapping[str, bytes], *,
               check: Callable[[Mapping[str, pathlib.Path]], None],
               served: Served | None = None,
               now: datetime.datetime | None = None) -> tuple[str, ...]:
    """Save several files of one step together (spec 7.1): refuse any that
    changed outside the wizard; write each ``<name>.part``; run ``check`` on
    the ``.part`` paths (a raise removes them and passes up); back up every
    old file; ``os.replace`` each into place; remember the new digests.
    Returns the names written."""
    folder = pathlib.Path(folder)
    served = SERVED if served is None else served
    now = datetime.datetime.now() if now is None else now
    with _SAVE_LOCK:
        targets = {name: resolve_inside(folder, name) for name in contents}
        for name in contents:
            served.check(folder, name)
            _backup_folder(folder, name)  # a planted link is refused before anything is written
        parts: dict[str, pathlib.Path] = {}
        try:
            for name, data in contents.items():
                part = targets[name].with_name(targets[name].name + PART_SUFFIX)
                part.parent.mkdir(parents=True, exist_ok=True)
                write_new(part, data)
                parts[name] = part
            check(dict(parts))
            for name in contents:
                backup(folder, name, now=now)
        except BaseException:
            for part in parts.values():
                part.unlink(missing_ok=True)
            raise
        for name in contents:
            replace_from_part(parts[name], targets[name])
        for name in contents:
            served.remember(folder, name)
    return tuple(contents)


# --- Recent projects ----------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RecentEntry:
    folder: str
    exists: bool


def default_recent_file() -> pathlib.Path:
    """``~/.leed-daylight/recent.json``. Tests point this into their own
    temporary folder (``tests/conftest.py``); callers look it up through
    this module at the moment of use."""
    return pathlib.Path.home() / ".leed-daylight" / "recent.json"


def _recent_folders(path) -> list[str]:
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    folders: list[str] = []
    for item in data:
        if isinstance(item, str) and os.path.isabs(item) and item not in folders:
            folders.append(item)
    return folders[:RECENT_MAX]


def _write_recent(path, folders: list[str]) -> None:
    """Readable by its owner only (``0600`` in a ``0700`` folder when new),
    written through a ``.part`` file created new, never through a link."""
    path = pathlib.Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    part = path.with_name(path.name + PART_SUFFIX)
    write_new(part, (json.dumps(folders[:RECENT_MAX], indent=2) + "\n").encode("utf-8"), mode=0o600)
    replace_from_part(part, path)


def read_recent(path) -> list[RecentEntry]:
    """The recent folders, newest first; a missing or malformed file is empty."""
    return [RecentEntry(folder, pathlib.Path(folder).is_dir()) for folder in _recent_folders(path)]


def add_recent(path, folder: pathlib.Path) -> None:
    """Put ``folder`` (made absolute) first, with no repeat, at most ten."""
    folder = str(pathlib.Path(folder).absolute())
    _write_recent(path, [folder] + [f for f in _recent_folders(path) if f != folder])


def remove_recent(path, folder: str) -> None:
    """Take one folder off the list."""
    _write_recent(path, [f for f in _recent_folders(path) if f != folder])
