# tests/mcp/test_mcp_arguments.py
"""Per-action arguments, folder containment and case names (spec 6)."""
import pytest

from leed_mcp.arguments import Action, check_action, check_case, resolve_folder
from leed_mcp.errors import Refused

TABLE = {"status": Action(required=frozenset({"folder"})),
         "files": Action(required=frozenset({"folder"}), optional=frozenset({"limit"}))}


def _codes(refused):
    return [(a["code"], a.get("field")) for a in refused.value.result.structured_content["advisories"]]


def test_an_argument_the_action_does_not_take_is_unknown():
    with pytest.raises(Refused) as caught:
        check_action("project", "status", {"folder": "/x", "limit": 5}, TABLE)
    assert _codes(caught) == [("UNKNOWN_ARGUMENT", "limit")]
    assert "project(action='status') does not take 'limit'; it takes: folder." in \
        caught.value.result.structured_content["reason"]


def test_a_required_argument_left_out_is_missing():
    with pytest.raises(Refused) as caught:
        check_action("project", "files", {"folder": None, "limit": None}, TABLE)
    assert _codes(caught) == [("ARGUMENT_MISSING", "folder")]


def test_a_complete_call_passes():
    assert check_action("project", "files", {"folder": "/x", "limit": 5}, TABLE) is None


def test_folder_outside_every_root_and_without_roots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(Refused) as caught:
        resolve_folder(str(tmp_path / "other"), (root.resolve(),))
    assert _codes(caught) == [("PATH_OUTSIDE_ROOTS", "folder")]
    with pytest.raises(Refused) as caught:
        resolve_folder(str(root), ())
    assert _codes(caught) == [("ROOTS_NOT_CONFIGURED", "folder")]


def test_a_missing_folder_is_file_missing_unless_it_may_be_new(tmp_path):
    root = tmp_path.resolve()
    with pytest.raises(Refused) as caught:
        resolve_folder(str(root / "new"), (root,))
    assert _codes(caught) == [("FILE_MISSING", "folder")]
    assert resolve_folder(str(root / "new"), (root,), must_exist=False) == root / "new"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "a:b", "a\x01b"])
def test_an_unsafe_case_name_is_input_invalid(bad):
    with pytest.raises(Refused) as caught:
        check_case(bad)
    assert _codes(caught) == [("INPUT_INVALID", "case")]


def test_a_safe_case_name_passes():
    assert check_case("base_case") == "base_case"


def test_dot_dot_segments_and_a_trailing_slash_resolve_inside_the_root(tmp_path):
    root = tmp_path.resolve()
    (root / "a").mkdir()
    (root / "b").mkdir()
    assert resolve_folder(f"{root}/a/../b/", (root,)) == root / "b"
    with pytest.raises(Refused) as caught:
        resolve_folder(f"{root}/b/../..", (root,))
    assert _codes(caught) == [("PATH_OUTSIDE_ROOTS", "folder")]


def test_the_server_reads_its_roots_from_the_environment(tmp_path, monkeypatch):
    from leed_mcp.server import build_server
    monkeypatch.setenv("LEED_MCP_ROOTS", str(tmp_path))
    assert build_server().roots == (tmp_path.resolve(),)
    assert build_server(roots=()).roots == ()
