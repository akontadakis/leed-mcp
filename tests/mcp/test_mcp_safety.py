# tests/mcp/test_mcp_safety.py
"""Output safety (spec 9): control characters removed, one field capped."""
from leed_mcp.envelope import LeedResult, emit
from leed_mcp.safety import FIELD_MAX, clean_text, sanitize


def test_control_characters_go_and_newline_and_tab_stay():
    assert clean_text("a\x1b[31mb\x00c\x7fd\x85e\r\nf\tg") == "a[31mbcde\nf\tg"


def test_a_long_field_is_cut_and_the_cut_is_marked():
    text = "x" * (FIELD_MAX + 5)
    assert clean_text(text) == "x" * FIELD_MAX + " [cut: 5 more characters]"
    assert clean_text("x" * FIELD_MAX) == "x" * FIELD_MAX


def test_sanitize_walks_keys_lists_and_leaves_numbers():
    assert sanitize({"k\x07": ["v\x08", 1, 2.5, None, True, {"n": "\x1b"}]}) == {
        "k": ["v", 1, 2.5, None, True, {"n": ""}]}


def test_emit_sanitizes_every_string():
    class _Note(LeedResult):
        note: str
    result = emit(_Note(note="room\x1b]0;title\x07 A"))
    assert result.structured_content == {"note": "room]0;title A"}
    assert "\x1b" not in result.content[0].text


def test_keys_that_differ_only_by_a_control_character_both_stay():
    assert sanitize({"a\x07": 1, "a": 2}) == {"a": 1, "a [2]": 2}
