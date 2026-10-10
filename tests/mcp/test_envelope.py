# tests/mcp/test_envelope.py
"""The one place a tool result is built (spec 9: refusal shape)."""
import json
from typing import Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import RootModel

from leed_mcp.envelope import (
    Advisory, LeedResult, Refusal, emit, one_of_schema, refuse, refusal, refuse_missing,
)


class _Base(LeedResult):
    action: str


class _Echo(_Base):
    action: Literal["echo"]
    text: str


class _Response(RootModel[_Echo | Refusal]):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return one_of_schema(super().model_json_schema(*args, **kwargs), _Base)


def test_emit_carries_the_payload_twice():
    result = emit(_Echo(action="echo", text="hi"))
    assert result.is_error is False
    assert result.structured_content == {"action": "echo", "text": "hi"}
    assert json.loads(result.content[0].text) == result.structured_content


def test_empty_advisories_are_omitted_and_a_null_field_is_dropped():
    with_one = emit(_Echo(action="echo", text="x", advisories=[
        Advisory(code="RESULT_TRUNCATED", severity="warn", message="cut")]))
    assert with_one.structured_content["advisories"] == [
        {"code": "RESULT_TRUNCATED", "severity": "warn", "message": "cut"}]
    assert "advisories" not in emit(_Echo(action="echo", text="x")).structured_content


def test_a_refusal_is_a_tool_error_with_its_code():
    result = refusal("LOCK_HELD", "a job holds the folder.", field="folder")
    assert result.is_error is True
    assert result.structured_content == {
        "refused": True,
        "reason": "a job holds the folder.",
        "advisories": [{"code": "LOCK_HELD", "severity": "error", "field": "folder",
                        "message": "a job holds the folder."}],
    }


def test_refuse_missing_names_each_absent_argument():
    result = refuse_missing("probe(action='echo')", text=None, count=3, folder=None)
    codes = [(a["code"], a["field"]) for a in result.structured_content["advisories"]]
    assert codes == [("ARGUMENT_MISSING", "folder"), ("ARGUMENT_MISSING", "text")]
    assert refuse_missing("probe", text="x") is None


def test_nan_is_never_emitted():
    class _F(LeedResult):
        value: float
    with pytest.raises(ValueError):
        emit(_F(value=float("nan")))


def test_the_union_schema_is_an_object_with_one_of_and_a_refusal_branch():
    schema = _Response.model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert "oneOf" in schema and "anyOf" not in schema
    validator = Draft202012Validator(schema)
    validator.validate(emit(_Echo(action="echo", text="x")).structured_content)
    validator.validate(refuse("no.").structured_content)
    validator.validate(refusal("LOCK_HELD", "no.").structured_content)
    assert list(validator.iter_errors({"action": "echo"}))  # text missing
    assert list(validator.iter_errors({"refused": False, "reason": "x"}))
