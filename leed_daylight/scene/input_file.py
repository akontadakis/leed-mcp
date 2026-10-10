# leed_daylight/scene/input_file.py
"""Bounded reads of the project's input files (security review L2).

``project.yaml``, the CSV files and the scene files come from whoever sent
the folder. Each is read here: only a plain file (a link is followed, but a
FIFO, a device or a socket is refused at once, never waited on), and
refused when it holds more than ``INPUT_FILE_MAX`` bytes, so a crafted file
can neither block a command or the wizard nor fill the memory. The
refusal is one plain line (``InputFileError``, a ``ValueError``).
"""
from __future__ import annotations

import errno
import io
import os
import pathlib
import stat
import sys

#: The most bytes one input file may hold: far more than any real
#: ``project.yaml``, CSV file or scene file.
INPUT_FILE_MAX = 256 * 1024 * 1024
#: Opening an input file: never waiting on a FIFO; bytes as they are on Windows.
_OPEN = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


class InputFileError(ValueError):
    """An input file that is not a plain file, or is too large."""


def open_input(path, *, platform: str | None = None) -> tuple[io.BufferedReader, int]:
    """Open the input file ``path`` for reading bytes: ``(handle, limit)``,
    ``limit`` being ``INPUT_FILE_MAX`` now. ``FileNotFoundError`` when there
    is none, ``IsADirectoryError`` for a folder, ``InputFileError`` for any
    other non-plain file (never waited on) or one whose size is over the
    limit. The caller closes the handle and refuses, with ``too_large``,
    more than ``limit`` bytes read (a file may grow after the check).
    Windows refuses to open a folder (``PermissionError``); that is an
    ``IsADirectoryError`` here too."""
    platform = platform or sys.platform
    path = pathlib.Path(path)
    limit = INPUT_FILE_MAX
    try:
        fd = os.open(path, _OPEN)
    except PermissionError:
        if platform == "win32" and os.path.isdir(path):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path)) from None
        raise
    handle = os.fdopen(fd, "rb")
    try:
        mode = os.fstat(fd).st_mode
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
        if not stat.S_ISREG(mode):
            raise InputFileError(f"{path}: is not a plain file; it is not read")
        if os.fstat(fd).st_size > limit:
            raise too_large(path, limit)
    except BaseException:
        handle.close()
        raise
    return handle, limit


def too_large(path, limit: int) -> InputFileError:
    """The refusal of an input file of more than ``limit`` bytes."""
    return InputFileError(
        f"{pathlib.Path(path)}: is larger than {limit} bytes; an input file this large is not read"
    )


def read_input_bytes(path) -> bytes:
    """The bytes of the input file ``path``, opened by ``open_input`` (its
    refusals)."""
    handle, limit = open_input(path)
    with handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise too_large(path, limit)
    return data


def read_input_text(path, encoding: str | None = None, errors: str | None = None) -> str:
    """``Path.read_text(encoding, errors)`` of the input file ``path``,
    read as ``read_input_bytes`` reads it (the same universal newlines)."""
    data = read_input_bytes(path)
    with io.TextIOWrapper(io.BytesIO(data), encoding=encoding, errors=errors) as text:
        return text.read()
