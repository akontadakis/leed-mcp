# leed_mcp/paths.py
"""Where the server may read and write (spec 5.3 and 10).

Adapted from rad_mcp (``src/rad_mcp/paths.py``): the containment rule is
the same (resolve first, then check, so a symlink leading out is refused);
the default-root half is dropped; the variable is ``LEED_MCP_ROOTS``.
An empty set of roots refuses every path: an unconfigured server is inert.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from leed_daylight.drivers._folder import repository_root

__all__ = [
    "PathOutsideRoots", "REPO_ROOT", "ROOTS_ENV_VAR", "RootsNotConfigured",
    "contain", "inside_root", "roots_from_env",
]

ROOTS_ENV_VAR = "LEED_MCP_ROOTS"
#: This repository, also when installed with ``--no-editable`` (``repository_root``).
REPO_ROOT = repository_root()

logger = logging.getLogger(__name__)


class PathOutsideRoots(ValueError):
    """A path resolved outside every allowed root, or was not absolute."""


class RootsNotConfigured(RuntimeError):
    """No usable root is configured, so no project path can be accepted."""


def _overlaps(root: Path, repository: Path) -> str | None:
    if inside_root(repository, root):
        return "it contains this repository"
    if inside_root(root, repository):
        return "it is inside this repository"
    return None


def roots_from_env(environ: Mapping[str, str] | None = None, *,
                   repository: Path = REPO_ROOT) -> tuple[Path, ...]:
    """The usable roots from ``LEED_MCP_ROOTS``, resolved, in order.

    Each entry must be absolute (``~`` is not expanded) and an existing
    folder, and must neither contain nor sit inside this repository. A bad
    entry is dropped with one warning on stderr; the rest still count.
    """
    raw = (os.environ if environ is None else environ).get(ROOTS_ENV_VAR, "")
    repository = repository.resolve()
    roots: list[Path] = []
    for part in (p.strip() for p in raw.split(os.pathsep)):
        if not part:
            continue
        path = Path(part)
        if not path.is_absolute():
            logger.warning("%s entry %r dropped: it is not absolute", ROOTS_ENV_VAR, part)
            continue
        if not path.is_dir():
            logger.warning("%s entry %r dropped: it is not an existing folder", ROOTS_ENV_VAR, part)
            continue
        resolved = path.resolve()
        if (why := _overlaps(resolved, repository)) is not None:
            logger.warning("%s entry %r dropped: %s", ROOTS_ENV_VAR, part, why)
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def inside_root(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` or lies below it. Besides the written
    form it compares each existing ancestor of ``path`` with ``root`` by
    device and inode, so a different spelling of the case on a case-folding
    file system (macOS, Windows) does not slip past."""
    if path == root or root in path.parents:
        return True
    try:
        root_info = os.stat(root)
    except OSError:
        return False
    for ancestor in (path, *path.parents):
        try:
            info = os.stat(ancestor)
        except OSError:
            continue
        # Some network, FUSE and FAT volumes report inode 0 for everything: no match can be told from it.
        if info.st_ino != 0 and os.path.samestat(info, root_info):
            return True
    return False


def _refuse_unresolved_link(resolved: Path, text: str) -> None:
    """Raise ``RuntimeError`` when a link is still left in ``resolved``.
    On macOS and Linux ``Path.resolve`` itself raises ``RuntimeError`` for a
    link loop; on Windows it gives the path back with the link left in it
    (a loop or a link that cannot be followed), which would then pass as a
    path inside its root. Resolved, no part of a path is a link."""
    for part in (resolved, *resolved.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise RuntimeError(f"{text!r} cannot be resolved: {part} is a link that cannot be followed")


def contain(candidate: str | Path, roots: Sequence[Path]) -> Path:
    """Resolve ``candidate`` and return it, or raise if it is outside every root.

    Checked after resolution, so a symlink inside a root that points
    outside it is refused.
    """
    if not roots:
        raise RootsNotConfigured(
            f"No usable root is configured. Set {ROOTS_ENV_VAR} to the folders this server may "
            "read and write."
        )
    text = str(candidate)
    if "\0" in text:
        raise PathOutsideRoots("The path contains a null byte.")
    path = Path(text)
    if not path.is_absolute():
        raise PathOutsideRoots(f"Paths must be absolute; got {text!r}.")
    resolved = path.resolve()
    _refuse_unresolved_link(resolved, text)
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise PathOutsideRoots(
        f"{resolved} is outside every allowed root ({', '.join(str(r) for r in roots)})."
    )
