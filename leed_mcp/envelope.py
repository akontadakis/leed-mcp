# leed_mcp/envelope.py
"""Building tool results: the one place a ``CallToolResult`` is constructed.

Copied from rad_mcp (``src/rad_mcp/envelope.py``) and trimmed; see PORTED.md.
Two rules from MCP revision 2026-07-28 live here:

- **A declared output schema binds.** *"If an output schema is provided:
  Servers MUST provide structured results that conform to this schema."*
  Every tool returns a model in this module's shape, and every tool's
  output union includes ``Refusal``, so a refusal validates too.
- **A refusal is a tool execution error.** It sets ``isError: true`` and
  keeps ``refused``, ``reason`` and the coded advisories in
  ``structuredContent``, so the model can correct itself.
"""
from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict

from leed_mcp.safety import sanitize

__all__ = [
    "Advisory", "LeedResult", "Refusal", "Severity",
    "emit", "one_of_schema", "refuse", "refusal", "refuse_missing",
]

Severity = Literal["info", "warn", "error"]


class Advisory(BaseModel):
    """One finding about the call or a value in its result.

    `code` is stable and machine-readable; branch on it, not on `message`.
    `field` names the argument or result field the finding is about.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    field: str | None = None
    message: str


class LeedResult(BaseModel):
    """Base of every tool result. Keys in ``OMIT_WHEN_NONE`` disappear when None."""

    model_config = ConfigDict(extra="forbid")

    advisories: list[Advisory] | None = None

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"advisories"})

    def payload(self) -> dict[str, Any]:
        """The `structuredContent` for this result."""
        data = self.model_dump(mode="json")
        for key in type(self).OMIT_WHEN_NONE:
            if data.get(key) is None:
                data.pop(key, None)
        for advisory in data.get("advisories") or []:
            if advisory.get("field") is None:
                advisory.pop("field", None)
        return data


class Refusal(LeedResult):
    """The call was refused, and the tool result has `isError: true`.

    `reason` says why and what to change; `advisories` carries the coded cause.
    """

    refused: Literal[True] = True
    reason: str


def emit(result: LeedResult, *, is_error: bool = False,
         meta: dict[str, Any] | None = None) -> CallToolResult:
    """Wrap a result model as a tool result: `structuredContent` plus the same
    JSON as text (*"a tool that returns structured content SHOULD also return
    the serialized JSON in a TextContent block"*). ``allow_nan=False``: NaN
    and Infinity are not JSON, so a result holding one is a bug, raised here.
    Every string is passed through ``safety.sanitize`` (spec 9, output safety)."""
    payload = sanitize(result.payload())
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2, allow_nan=False))],
        structured_content=payload,
        is_error=is_error,
        meta=meta,
    )


def refuse(reason: str, advisories: list[Advisory] | None = None) -> CallToolResult:
    """The refusal envelope: `isError: true` and the payload to self-correct."""
    return emit(Refusal(reason=reason, advisories=advisories or None), is_error=True)


def refusal(code: str, reason: str, *, field: str | None = None) -> CallToolResult:
    """A refusal with one coded `error` advisory whose message is the reason."""
    return refuse(reason, [Advisory(code=code, severity="error", field=field, message=reason)])


def refuse_missing(call: str, *, hint: str | None = None, **arguments: Any) -> CallToolResult | None:
    """The refusal for required arguments that were not given, or None.

    One `ARGUMENT_MISSING` advisory per absent argument, each with its `field`.
    """
    absent = sorted(key for key, value in arguments.items() if value is None)
    if not absent:
        return None
    reason = f"{call} requires {', '.join(absent)}." + (f" {hint}" if hint else "")
    return refuse(reason, [
        Advisory(code="ARGUMENT_MISSING", severity="error", field=name, message=reason)
        for name in absent
    ])


def one_of_schema(schema: dict[str, Any], base: type[BaseModel]) -> dict[str, Any]:
    """Rewrite a `RootModel[Union[...]]` schema into an object with `oneOf`.

    `anyOf` becomes `oneOf` (exactly one branch must match), and the shared
    base's properties are inlined at the top with `type: "object"` (the
    legacy 2025-11-25 era requires an object root). The base must not be one
    of the branches, and its discriminator must be typed wider (`str`) than
    any branch's `Literal`, or its constants would bind every branch.
    """
    branches = schema.pop("anyOf", None)
    if branches is None:  # pragma: no cover - a one-member union is a caller bug
        raise ValueError("one_of_schema expects a union root model; got no 'anyOf'.")
    base_schema = base.model_json_schema()
    definitions = schema.setdefault("$defs", {})
    definitions.update(base_schema.pop("$defs", {}))
    schema["type"] = "object"
    schema["properties"] = base_schema.get("properties", {})
    schema["oneOf"] = branches
    return schema
