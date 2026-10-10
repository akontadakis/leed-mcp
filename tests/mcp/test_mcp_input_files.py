# tests/mcp/test_mcp_input_files.py
"""The write path every inputs write takes (spec 6.3 rules 1-4, spec 10, N1)."""
import datetime
import hashlib
import json
import os
import stat
import sys

import pytest
import yaml

from conftest import symlink_or_skip
from leed_daylight.engine.jobs import LockHeld, UnsafePath, acquire_lock, release_lock
from leed_daylight.scene import input_file
from leed_mcp.adapters import input_files as io_
from leed_mcp.errors import Refused, refusal_for


def _project(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "project.yaml").write_text(
        'weather:\n  file: "weather/site.wea"\nstandard:\n  credit_rules: "credit_rules.yaml"\n')
    return folder


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(folder, name, expected, data=b"new\n", check=lambda part: None):
    return io_.write_file(folder, name, expected, lambda old, raw: data, check)


def _code(refused: Refused) -> tuple[str, str]:
    content = refused.result.structured_content
    return content["advisories"][0]["code"], content["reason"]


def test_a_new_file_is_written_when_expected_is_none(tmp_path):
    folder = _project(tmp_path)
    written = _write(folder, "notes.csv", None)
    assert (folder / "notes.csv").read_bytes() == b"new\n"
    assert written == io_.Written(folder / "notes.csv", _sha(b"new\n"), None)


def test_a_replace_backs_up_the_old_file(tmp_path):
    folder = _project(tmp_path)
    (folder / "notes.csv").write_bytes(b"old\n")
    written = _write(folder, "notes.csv", _sha(b"old\n"))
    assert (folder / "notes.csv").read_bytes() == b"new\n"
    assert written.backup.parent == folder / "_run" / "backup"
    assert written.backup.name.startswith("notes.csv.") and written.backup.read_bytes() == b"old\n"


def test_a_file_in_a_sub_folder_is_backed_up_under_the_same_sub_folder(tmp_path):
    folder = _project(tmp_path)
    (folder / "weather").mkdir()
    (folder / "weather" / "site.wea").write_bytes(b"old\n")
    written = _write(folder, "weather/site.wea", _sha(b"old\n"))
    assert written.backup.parent == folder / "_run" / "backup" / "weather"
    assert written.backup.read_bytes() == b"old\n"


@pytest.mark.parametrize("on_disk, expected, words", [
    (b"old\n", _sha(b"other\n"), "changed on disk since it was read"),
    (b"old\n", None, "already exists"),
    (None, _sha(b"old\n"), "does not exist"),
])
def test_a_wrong_expected_sha256_is_refused_and_nothing_changes(tmp_path, on_disk, expected, words):
    folder = _project(tmp_path)
    if on_disk is not None:
        (folder / "notes.csv").write_bytes(on_disk)
    with pytest.raises(Refused) as caught:
        _write(folder, "notes.csv", expected)
    code, reason = _code(caught.value)
    assert code == "INPUT_CHANGED_ON_DISK" and words in reason and "wizard" not in reason
    if on_disk is None:
        assert not (folder / "notes.csv").exists()
    else:
        assert (folder / "notes.csv").read_bytes() == on_disk


def test_a_failed_check_leaves_the_file_and_no_part(tmp_path):
    folder = _project(tmp_path)
    (folder / "notes.csv").write_bytes(b"old\n")

    def check(part):
        assert part.name == "notes.csv.part" and part.read_bytes() == b"new\n"
        raise ValueError("notes.csv: bad row")

    with pytest.raises(ValueError, match="bad row"):
        _write(folder, "notes.csv", _sha(b"old\n"), check=check)
    assert (folder / "notes.csv").read_bytes() == b"old\n"
    assert not (folder / "notes.csv.part").exists() and not (folder / "_run" / "backup").exists()


def test_a_link_at_the_file_is_refused_and_its_target_is_untouched(tmp_path):
    folder = _project(tmp_path)
    (folder / "real.csv").write_bytes(b"old\n")
    symlink_or_skip(folder / "notes.csv", folder / "real.csv")
    with pytest.raises(UnsafePath, match="symbolic link"):
        _write(folder, "notes.csv", _sha(b"old\n"))
    assert (folder / "real.csv").read_bytes() == b"old\n"


def test_a_running_job_holds_the_folder_lock_and_the_write_is_refused(tmp_path):
    folder = _project(tmp_path)
    info = acquire_lock(folder, "run")
    try:
        with pytest.raises(LockHeld):
            _write(folder, "notes.csv", None)
    finally:
        release_lock(folder, info)
    assert not (folder / "notes.csv").exists()
    assert refusal_for(LockHeld("busy")).structured_content["advisories"][0]["code"] == "LOCK_HELD"


@pytest.mark.parametrize("key, value", [
    ("weather", '{file: "../outside.wea"}'),
    ("grid", '{points_file: "/elsewhere/points.csv"}'),
    ("atlas", '{drawings: [{image: "../plan.png"}]}'),
    ("scene", '{files: ["model/a.rad", "../b.rad"]}'),
])
def test_a_project_yaml_path_outside_the_folder_refuses_every_write(tmp_path, key, value):
    folder = _project(tmp_path)
    text = (folder / "project.yaml").read_text()
    if key == "weather":
        text = text.replace('  file: "weather/site.wea"\n', "").replace("weather:\n", "")
    (folder / "project.yaml").write_text(text + f"{key}: {value}\n")
    with pytest.raises(Refused) as caught:
        _write(folder, "notes.csv", None)
    code, reason = _code(caught.value)
    assert code == "PATH_OUTSIDE_ROOTS" and "outside the project folder" in reason
    assert not (folder / "notes.csv").exists()


def test_a_link_inside_the_folder_that_leads_out_counts_as_outside(tmp_path):
    folder = _project(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    symlink_or_skip(folder / "weather", tmp_path / "elsewhere")
    with pytest.raises(Refused) as caught:
        _write(folder, "notes.csv", None)
    assert _code(caught.value)[0] == "PATH_OUTSIDE_ROOTS"


def test_an_unreadable_project_yaml_refuses_other_writes(tmp_path):
    folder = _project(tmp_path)
    (folder / "project.yaml").write_text("weather: [unclosed\n")
    with pytest.raises(Refused) as caught:
        _write(folder, "notes.csv", None)
    code, reason = _code(caught.value)
    assert code == "INPUT_INVALID" and reason.startswith("project.yaml cannot be read")


def test_the_source_log_gets_one_private_line_per_update(tmp_path):
    folder = _project(tmp_path)
    for n in (1, 2):
        path = io_.log_source(folder, name="project.yaml", keys=["site.latitude_N"], unset=[],
                              source=f"typed by the user in chat, message {n}")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert path == folder / "_run" / "input_log.jsonl"
    assert [line["source"] for line in lines] == ["typed by the user in chat, message 1",
                                                  "typed by the user in chat, message 2"]
    assert lines[0]["set"] == ["site.latitude_N"] and lines[0]["unset"] == []
    assert datetime.datetime.fromisoformat(lines[0]["time"]).utcoffset() == datetime.timedelta(0)
    if sys.platform != "win32":  # POSIX modes
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_jsonable_makes_yaml_values_json():
    loaded = {1: datetime.date(2026, 3, 21), "x": [float("nan"), float("inf"), 0.5], "y": {"z": None}}
    assert io_.jsonable(loaded) == {"1": "2026-03-21", "x": [None, None, 0.5], "y": {"z": None}}


def _laughs(levels: int = 9) -> str:
    """A YAML 'billion laughs': each level holds ten aliases of the one before."""
    lines = ['l0: &l0 ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]']
    lines += [f"l{n}: &l{n} [{', '.join([f'*l{n - 1}'] * 10)}]" for n in range(1, levels + 1)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("text, words", [
    (_laughs(), "more than 100000 nodes"),
    ("a: &a [*a]\n", "recursive alias"),
    ("a: &a {b: *a}\n", "recursive alias"),
    ("[" * 5000 + "]" * 5000 + "\n", "nested too deeply"),
])
def test_load_yaml_refuses_an_expanding_or_recursive_document(text, words):
    with pytest.raises(io_.YAMLTooLarge, match=words) as caught:
        io_.load_yaml(text)
    assert isinstance(caught.value, yaml.YAMLError)


def test_load_yaml_keeps_ordinary_anchors_and_the_cap_counts_expanded_nodes():
    assert io_.load_yaml("base: &b {x: 1}\nother: *b\nlist: [*b, *b]\n") == {
        "base": {"x": 1}, "other": {"x": 1}, "list": [{"x": 1}, {"x": 1}]}
    assert io_.load_yaml("") is None
    assert len(io_.load_yaml(_laughs(3))["l3"]) == 10
    with pytest.raises(io_.YAMLTooLarge):
        io_.load_yaml(_laughs(5))


def test_an_expanding_project_yaml_refuses_other_writes_and_its_own(tmp_path):
    folder = _project(tmp_path)
    (folder / "project.yaml").write_text(_laughs())
    with pytest.raises(Refused) as caught:
        _write(folder, "notes.csv", None)
    code, reason = _code(caught.value)
    assert code == "INPUT_INVALID" and "more than 100000 nodes" in reason
    with pytest.raises(yaml.YAMLError):
        _write(folder, "project.yaml", _sha((folder / "project.yaml").read_bytes()))
    assert not (folder / "notes.csv").exists()


def test_jsonable_makes_bytes_null_and_sets_sorted_lists():
    assert io_.jsonable({"b": b"\x00\x01", "s": {3, 1, 2}, "f": frozenset({"b", "a"}), "m": {1, "a"}}) == {
        "b": None, "s": [1, 2, 3], "f": ["a", "b"], "m": [1, "a"]}


def test_read_bytes_opens_without_following_a_link_even_past_the_lstat_check(tmp_path, monkeypatch):
    folder = _project(tmp_path)
    (tmp_path / "real.csv").write_bytes(b"secret\n")
    symlink_or_skip(folder / "notes.csv", tmp_path / "real.csv")
    monkeypatch.setattr(os.path, "islink", lambda path: False)  # the link appears after any lstat check
    with pytest.raises(UnsafePath, match="symbolic link"):
        io_.read_bytes(folder, "notes.csv")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_read_bytes_refuses_a_fifo(tmp_path):
    folder = _project(tmp_path)
    os.mkfifo(folder / "pipe.csv")
    with pytest.raises(UnsafePath, match="not a plain file"):
        io_.read_bytes(folder, "pipe.csv")


def test_read_bytes_refuses_a_folder_and_a_file_over_the_cap(tmp_path, monkeypatch):
    folder = _project(tmp_path)
    (folder / "sub.csv").mkdir()
    (folder / "big.csv").write_bytes(b"0123456789" * 2)
    with pytest.raises(UnsafePath, match="not a plain file"):
        io_.read_bytes(folder, "sub.csv")
    assert io_.read_bytes(folder, "absent.csv") is None
    assert io_.read_bytes(folder, "project.yaml").startswith(b"weather:")
    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", 10)
    with pytest.raises(input_file.InputFileError, match="larger than 10 bytes"):
        io_.read_bytes(folder, "big.csv")


def test_an_oversized_project_yaml_is_path_unsafe_for_a_write_not_input_invalid(tmp_path, monkeypatch):
    folder = _project(tmp_path)
    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", 10)
    with pytest.raises(input_file.InputFileError) as caught:
        _write(folder, "notes.csv", None)
    assert refusal_for(caught.value).structured_content["advisories"][0]["code"] == "PATH_UNSAFE"
    assert not (folder / "notes.csv").exists()


def test_an_oversized_project_yaml_is_not_ignored_when_naming_credit_rules(tmp_path, monkeypatch):
    from leed_mcp.adapters import input_read
    folder = _project(tmp_path)
    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", 10)
    with pytest.raises(input_file.InputFileError):
        input_read.credit_rules_name(folder)


@pytest.mark.parametrize("make", [
    lambda path: FileNotFoundError(2, "No such file or directory", str(path)),
    lambda path: OSError(2, "No such file or directory", str(path), str(path.parent / "other")),
    lambda path: ValueError(f"{os.path.realpath(path)}: bad row"),
    lambda path: ValueError(f"{path}: bad row"),
])
def test_as_file_never_names_the_temporary_folder(tmp_path, make):
    folder = _project(tmp_path)
    with pytest.raises((OSError, ValueError)) as caught:
        with io_.as_file(folder, "notes.csv", b"x\n") as path:
            tmp = str(path.parent)
            real = os.path.realpath(tmp)
            raise make(path)
    text = str(caught.value)
    assert tmp not in text and real not in text and str(folder / "notes.csv") in text
    if isinstance(caught.value, ValueError):
        assert text == f"{folder / 'notes.csv'}: bad row"
