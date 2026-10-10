# leed_mcp/tasks.py
"""The Tasks extension ``io.modelcontextprotocol/tasks``, revision 2026-07-28
(spec 8.3; D8: written new, rad_mcp's ``tasks.py`` is not copied).

- ``run`` and ``deliver`` answer a request that declares the extension with a
  flat ``CreateTaskResult`` whose ``taskId`` is the job id; any other request
  gets the ``started`` branch as before. The job's record is written before
  either reply (``JobRunner.start``), so ``tasks/get`` resolves at once.
- A request declares only through its own
  ``_meta["io.modelcontextprotocol/clientCapabilities"].extensions``; earlier
  declarations do not count, and a legacy (2025-11-25) request never declares.
- ``tasks/get``, ``tasks/update`` and ``tasks/cancel`` are bound to 2026-07-28
  only, so a legacy connection gets -32601. A modern request that does not
  declare gets -32021; an unknown, expired or out-of-roots ``taskId`` -32602;
  a job record that cannot be read -32603, its state unchanged. Every task
  request reads the record through ``JobRunner``, which checks again that the
  job's folder is inside the roots.
- Task status notifications are declined: the SDK's ``subscriptions/listen``
  acknowledgement drops ``taskIds``, and ``decline_task_notifications`` answers
  -32021 to a request that asks for them without declaring.

Text from logs and refusals reaches a task outside ``envelope.emit`` (the
``statusMessage``), so it passes ``safety.clean_text`` here (spec 9).
"""
from __future__ import annotations

import datetime
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import pydantic
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY, INTERNAL_ERROR, INVALID_PARAMS, MISSING_REQUIRED_CLIENT_CAPABILITY,
    CallToolRequestParams, CallToolResult, ClientCapabilities, MissingRequiredClientCapabilityErrorData,
    RequestParams,
)

from leed_mcp.adapters.job_runner import JobRunner, JobView
from leed_mcp.adapters.job_state import RETENTION_S
from leed_mcp.envelope import emit
from leed_mcp.errors import in_worker
from leed_mcp.jobindex import JOB_ID
from leed_mcp.safety import clean_text
from leed_mcp.tools.run import Ended

__all__ = [
    "POLL_INTERVAL_MS", "STATUS", "TASKS", "TASK_TOOLS", "TTL_MS", "Tasks", "VERSIONS", "decline_task_notifications",
    "declares", "detailed_task",
]

#: The extension's identifier.
TASKS = "io.modelcontextprotocol/tasks"
#: The protocol versions the ``tasks/*`` methods are bound to (spec 8.3, Eras).
VERSIONS = frozenset({"2026-07-28"})
#: A task lives as long as its job record (spec 8.1, retention), in ms.
TTL_MS = RETENTION_S * 1000
#: How often a client should poll ``tasks/get``, in ms. Chosen, not measured.
POLL_INTERVAL_MS = 5000
#: The tools whose start may be answered with a task.
TASK_TOOLS = frozenset({"run", "deliver"})
#: Task status per job state (spec 8.2). ``expired`` has none: -32602.
STATUS = {"running": "working", "succeeded": "completed", "failed": "completed", "cancelled": "cancelled"}


class TaskParams(RequestParams):
    """``tasks/get`` and ``tasks/cancel``."""

    taskId: str  # noqa: N815 - the wire name


class UpdateTaskParams(TaskParams):
    """``tasks/update``. This server never asks for input, so
    ``inputResponses`` is accepted and ignored."""

    inputResponses: dict[str, Any] | None = None  # noqa: N815 - the wire name


def declares(meta: Mapping[str, Any] | None, version: str | None) -> bool:
    """Whether a request with this ``_meta`` at this protocol version
    declares the extension (spec 8.3). On SDK 2.3 ``ctx.meta`` is a plain
    dict keyed by the wire names (spec 13, N2)."""
    if version not in VERSIONS or not isinstance(meta, Mapping):
        return False
    capabilities = meta.get(CLIENT_CAPABILITIES_META_KEY)
    extensions = capabilities.get("extensions") if isinstance(capabilities, Mapping) else None
    return isinstance(extensions, Mapping) and TASKS in extensions


def _missing() -> MCPError:
    """-32021 with ``data.requiredCapabilities`` naming the extension."""
    data = MissingRequiredClientCapabilityErrorData(
        required_capabilities=ClientCapabilities(extensions={TASKS: {}}))
    return MCPError(code=MISSING_REQUIRED_CLIENT_CAPABILITY,
                    message=f"Missing required client capability: declare the {TASKS} extension on the request",
                    data=data.model_dump(by_alias=True, mode="json", exclude_none=True))


def _not_found(why: str = "task not found") -> MCPError:
    """-32602. The ``taskId`` is not echoed: it is the client's own value."""
    return MCPError(code=INVALID_PARAMS, message=f"Failed to retrieve task: {why}")


def _unreadable() -> MCPError:
    """-32603: the job record could not be read, or no longer fits; the
    task is unchanged."""
    return MCPError(code=INTERNAL_ERROR, message="Failed to retrieve task: its job record could not be read; "
                    "the task is unchanged, try again")


def _utc_text(text: str) -> str:
    """An ISO 8601 time as UTC with its offset; a time without an offset
    (an end record's ``ended``) is this computer's local time."""
    moment = datetime.datetime.fromisoformat(text).astimezone(datetime.timezone.utc)
    return moment.isoformat(timespec="seconds")


def _result(view: JobView) -> dict[str, Any]:
    """The completed task's result: the ``run`` or ``deliver`` tool result a
    job's end gives, ``isError`` true when it failed (spec 8.2)."""
    record, state = view.record, view.state
    result = emit(Ended(action=record.action, state=state.state, job_id=record.job_id, reason=state.reason,
                        finished=record.finished), is_error=state.state == "failed")
    return result.model_dump(by_alias=True, mode="json", exclude_none=True)


def detailed_task(view: JobView, note: str | None = None) -> dict[str, Any]:
    """The task of a job that is not expired (spec 8.2, 8.3). ``working``
    carries ``note`` (why a cancel did not stop it) or the latest progress
    line as ``statusMessage``. ``lastUpdatedAt`` is the job's end, or its
    start while it runs or when it left no end record."""
    record, state = view.record, view.state
    status = STATUS[state.state]
    task: dict[str, Any] = {
        "taskId": record.job_id, "status": status, "createdAt": record.created,
        "lastUpdatedAt": record.created if state.ended is None else _utc_text(state.ended),
        "ttlMs": TTL_MS, "pollIntervalMs": POLL_INTERVAL_MS}
    if status == "working":
        message = note or view.progress or f"the {record.action} job is running"
    elif status == "cancelled":
        message = f"the {record.action} job was cancelled"
    else:
        message = f"the {record.action} job {state.state}" + (f": {state.reason}" if state.reason else "")
        task["result"] = _result(view)
    task["statusMessage"] = clean_text(message)
    return task


def _code(refused: CallToolResult) -> str:
    """The code of a refusal's first advisory."""
    return refused.structured_content["advisories"][0]["code"]


class Tasks(Extension):
    """The extension over one server's jobs and roots."""

    identifier = TASKS

    def __init__(self, jobs: JobRunner, roots: Sequence[pathlib.Path]) -> None:
        self.jobs = jobs
        self.roots = tuple(roots)
        #: Why the latest ``tasks/cancel`` of a job did not stop it, by job id.
        self._notes: dict[str, str] = {}

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding(method="tasks/get", params_type=TaskParams, handler=self._get, protocol_versions=VERSIONS),
            MethodBinding(method="tasks/update", params_type=UpdateTaskParams, handler=self._update,
                          protocol_versions=VERSIONS),
            MethodBinding(method="tasks/cancel", params_type=TaskParams, handler=self._cancel,
                          protocol_versions=VERSIONS),
        ]

    async def _view(self, ctx: ServerRequestContext[Any, Any], task_id: str) -> JobView:
        """The job of ``task_id`` now, for a declaring request; the record
        is read in a worker thread through ``JobRunner.view``, which checks
        the folder against the roots again."""
        if not declares(ctx.meta, ctx.protocol_version):
            raise _missing()
        if not JOB_ID.fullmatch(task_id):
            raise _not_found()
        view = await in_worker(self.jobs.view, task_id, self.roots)
        if isinstance(view, CallToolResult):
            if _code(view) == "JOB_NOT_FOUND":
                raise _not_found()
            raise _unreadable()
        if view.state.state not in STATUS:
            raise _not_found("task has expired")
        return view

    async def _get(self, ctx: ServerRequestContext[Any, Any], params: TaskParams) -> dict[str, Any]:
        """A stored ``finished`` snapshot that no longer fits ``Finished``
        is a malformed record: -32603. The note of a refused stop is shown
        only while the task is ``working``, so it is dropped once it is not."""
        view = await self._view(ctx, params.taskId)
        try:
            task = detailed_task(view, self._notes.get(params.taskId))
        except pydantic.ValidationError:
            raise _unreadable() from None
        if task["status"] != "working":
            self._notes.pop(params.taskId, None)
        return {"resultType": "complete", **task}

    async def _update(self, ctx: ServerRequestContext[Any, Any], params: UpdateTaskParams) -> dict[str, Any]:
        await self._view(ctx, params.taskId)
        return {"resultType": "complete"}

    async def _cancel(self, ctx: ServerRequestContext[Any, Any], params: TaskParams) -> dict[str, Any]:
        """Acknowledged for every known task, even when the stop is refused:
        the task then stays ``working`` and its ``statusMessage`` says why."""
        await self._view(ctx, params.taskId)
        self._notes.pop(params.taskId, None)
        stopped = await in_worker(self.jobs.cancel, params.taskId, self.roots)
        if isinstance(stopped, CallToolResult):
            if _code(stopped) == "JOB_NOT_FOUND":
                raise _not_found()
            # Kept raw: ``detailed_task`` cleans the ``statusMessage`` once.
            self._notes[params.taskId] = f"not cancelled ({_code(stopped)}): {stopped.structured_content['reason']}"
        return {"resultType": "complete"}

    async def intercept_tool_call(self, params: CallToolRequestParams, ctx: ServerRequestContext[Any, Any],
                                  call_next: CallNext) -> HandlerResult:
        """The whole ``tools/call`` runs first (rate limit, arguments, the
        checks, the start); a declaring request's ``started`` branch from
        ``run`` or ``deliver`` then becomes a ``CreateTaskResult``."""
        result = await call_next(ctx)
        if params.name not in TASK_TOOLS or not declares(ctx.meta, ctx.protocol_version):
            return result
        if not isinstance(result, CallToolResult) or result.is_error:
            return result
        started = result.structured_content or {}
        if started.get("state") != "started":
            return result
        return {
            "resultType": "task", "taskId": started["job_id"], "status": "working",
            "statusMessage": clean_text(f"the {started['action']} job started on case {started['case']}"),
            "createdAt": started["created"], "lastUpdatedAt": started["created"], "ttlMs": TTL_MS,
            "pollIntervalMs": POLL_INTERVAL_MS}


async def decline_task_notifications(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    """``Server.middleware``: a modern ``subscriptions/listen`` that asks for
    ``taskIds`` without declaring the extension gets -32021 (spec 8.3). A
    declaring one goes on to the SDK, whose acknowledgement leaves
    ``taskIds`` out: this server sends no task notifications."""
    if ctx.method == "subscriptions/listen" and ctx.protocol_version in VERSIONS and isinstance(ctx.params, Mapping):
        notifications = ctx.params.get("notifications")
        if isinstance(notifications, Mapping) and "taskIds" in notifications and not declares(
                ctx.meta, ctx.protocol_version):
            raise _missing()
    return await call_next(ctx)
