# leed_daylight/engine/in_use.py
"""A file Excel or Word holds open, named as such (spec 2026-10-09 A1).

Windows refuses to replace, overwrite or delete a file another program has
open, with ``PermissionError``. Elsewhere the same error means a real
permission problem, so it is left as it is there.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import sys
from collections.abc import Iterator

__all__ = ["FileInUse", "explain_in_use", "remove_tree_explained", "replace_explained"]


class FileInUse(OSError):
    """Windows: another program (Excel, Word) holds the file open."""


#: The prefix of the owner file Office keeps beside an open document.
_OFFICE_OWNER = "~$"


def _message(path, *, folder: bool = False) -> str:
    name = pathlib.Path(path).name
    if folder:
        return f"a file inside {name} is open in another program; close it in Excel or Word, then try again"
    name = name.removeprefix(_OFFICE_OWNER)
    return f"{name} is open in another program; close {name} in Excel or Word, then try again"


@contextlib.contextmanager
def explain_in_use(path, *, folder: bool = False, platform: str | None = None) -> Iterator[None]:
    """Turn a Windows ``PermissionError`` in the body into ``FileInUse``,
    naming the file the error names (the second one, the target, when it
    names two, as a failed replace does), or else ``path``. ``folder``:
    ``path`` is a folder being moved, and the message says a file inside
    it is open."""
    platform = platform or sys.platform
    try:
        yield
    except PermissionError as exc:
        if platform != "win32":
            raise
        if folder:
            raise FileInUse(_message(path, folder=True)) from exc
        raise FileInUse(_message(exc.filename2 or exc.filename or path)) from exc


def replace_explained(part, target, *, platform: str | None = None) -> None:
    """``os.replace(part, target)``; ``FileInUse`` naming ``target`` when Windows refuses it because ``target`` is open."""
    try:
        os.replace(part, target)
    except PermissionError as exc:
        if (platform or sys.platform) != "win32":
            raise
        raise FileInUse(_message(target)) from exc


def _same_path(one, other) -> bool:
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def remove_tree_explained(folder, *, platform: str | None = None) -> None:
    """``shutil.rmtree(folder)``; ``FileInUse`` naming the open file that stopped it on Windows,
    or saying a file inside ``folder`` is open when the folder itself could not be removed.
    The folder may then be partly removed: the user closes the file and asks again."""
    try:
        shutil.rmtree(folder)
    except PermissionError as exc:
        if (platform or sys.platform) != "win32":
            raise
        blocked = exc.filename or folder
        if _same_path(blocked, folder):
            raise FileInUse(_message(folder, folder=True)) from exc
        raise FileInUse(_message(blocked)) from exc
