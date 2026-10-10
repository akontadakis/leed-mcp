# leed_mcp/errors.py
"""From a pipeline exception to a coded refusal (spec 9, "Mapping pipeline
errors"), and the worker-thread boundary every tool body runs behind.

The server makes its own checks first; what the pipeline still raises is
mapped by class, the most specific class first, because many pipeline
errors are ``ValueError`` subclasses. A class not in the table is left to
``LeedMCPServer.call_tool``, which refuses it as ``INTERNAL_ERROR`` with the
type only. The pipeline's message becomes ``reason``.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import anyio.to_thread
import yaml
from mcp_types import CallToolResult
from shapely.errors import ShapelyError

from leed_daylight.edit.files import ChangedOutside, PathOutside
from leed_daylight.engine.in_use import FileInUse
from leed_daylight.engine.jobs import JobError, LockHeld, UnsafePath
from leed_daylight.engine.radiance import RadianceError
from leed_daylight.scene.input_file import InputFileError
from leed_mcp.envelope import refusal
from leed_mcp.paths import PathOutsideRoots, RootsNotConfigured

__all__ = ["CHANGED_ON_DISK", "Refused", "in_worker", "one_line", "refusal_for"]


class Refused(Exception):
    """Raised inside a tool body to answer with ``result`` (a refusal)."""

    def __init__(self, result: CallToolResult) -> None:
        super().__init__("refused")
        self.result = result


#: (class, code), the most specific first. Spec 9's table, plus this
#: server's own two path errors. ``JobError`` maps to ``JOB_RUNNING`` (the
#: start case); ``JobRunner.cancel`` catches a stop's ``JobError`` itself
#: (``STOP_UNCONFIRMED``).
MAPPING: tuple[tuple[type[BaseException], str], ...] = (
    (RootsNotConfigured, "ROOTS_NOT_CONFIGURED"),
    (PathOutsideRoots, "PATH_OUTSIDE_ROOTS"),
    (LockHeld, "LOCK_HELD"),
    (ChangedOutside, "INPUT_CHANGED_ON_DISK"),
    (PathOutside, "PATH_OUTSIDE_ROOTS"),
    (UnsafePath, "PATH_UNSAFE"),
    (InputFileError, "PATH_UNSAFE"),
    (JobError, "JOB_RUNNING"),
    (RadianceError, "RADIANCE_FAILED"),
    (ValueError, "INPUT_INVALID"),
    (yaml.YAMLError, "INPUT_INVALID"),
    (ShapelyError, "INPUT_INVALID"),
    (FileInUse, "FILE_IN_USE"),
    (OSError, "IO_ERROR"),
)


#: Spec 13, N1: ``ChangedOutside`` says "changed outside the wizard; reload
#: the step", which names a program this server does not have. Its refusal
#: carries this reason instead, never the pipeline's text.
CHANGED_ON_DISK = ("the file changed on disk since it was read; read it again with "
                   "inputs(action='read') and send its new sha256 as expected_sha256.")
#: Classes whose refusal carries this server's reason, not the exception's.
_OWN_REASONS: dict[type[BaseException], str] = {ChangedOutside: CHANGED_ON_DISK}


def one_line(exc: BaseException) -> str:
    """The exception's message on one line, or its type name when empty."""
    return " ".join(str(exc).split()) or type(exc).__name__


def refusal_for(exc: BaseException) -> CallToolResult | None:
    """The coded refusal for ``exc``, or None for a class outside the table.
    The reason is the exception's message, or this server's own for a class
    in ``_OWN_REASONS``."""
    for cls, code in MAPPING:
        if isinstance(exc, cls):
            return refusal(code, _OWN_REASONS.get(cls) or one_line(exc))
    return None


async def in_worker(fn: Callable[..., CallToolResult], /, *args: Any, **kwargs: Any) -> CallToolResult:
    """Run a blocking tool body in a worker thread, off the event loop
    (spec 5.1). ``Refused`` answers with its refusal; a mapped pipeline
    exception answers with its coded refusal; anything else propagates to
    the server's ``INTERNAL_ERROR`` boundary."""
    try:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
    except Refused as refused:
        return refused.result
    except Exception as exc:  # noqa: BLE001 - mapped by class, or re-raised
        mapped = refusal_for(exc)
        if mapped is None:
            raise
        return mapped
