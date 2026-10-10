# tests/mcp/test_unknown_arguments.py
"""An argument a tool does not name is refused, never ignored (spec 6)."""
from leed_mcp.unknown_arguments import refuse_unknown_arguments, unknown_key_advisories


def test_known_arguments_pass():
    assert refuse_unknown_arguments("project", {"action": "status", "folder": "/x"},
                                    {"action", "folder"}) is None


def test_each_unknown_argument_gets_its_own_advisory_with_a_hint():
    result = refuse_unknown_arguments("project", {"action": "status", "foldr": "/x", "zzz": 1},
                                      {"action", "folder"})
    assert result.is_error is True
    advisories = result.structured_content["advisories"]
    assert [(a["code"], a["field"]) for a in advisories] == [
        ("UNKNOWN_ARGUMENT", "foldr"), ("UNKNOWN_ARGUMENT", "zzz")]
    assert "The closest name is 'folder'." in advisories[0]["message"]
    assert "Its arguments are: action, folder." in advisories[1]["message"]


def test_unknown_nested_keys_are_named_with_their_argument():
    advisories = unknown_key_advisories("values", {"grid.spacing_m": 1, "gird": 2},
                                        {"grid.spacing_m"}, field="values")
    assert [(a.code, a.field) for a in advisories] == [("UNKNOWN_ARGUMENT", "values")]
    assert "values has no key 'gird'." in advisories[0].message
