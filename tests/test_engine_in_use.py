"""A file another program holds open is named (spec 2026-10-09 A1)."""
import sys

import pytest

from leed_daylight.engine import in_use
from leed_daylight.engine.in_use import FileInUse, explain_in_use, remove_tree_explained, replace_explained


def test_on_windows_a_permission_error_says_close_the_file(tmp_path):
    with pytest.raises(FileInUse, match=r"close calc\.xlsm in Excel or Word, then try again"):
        with explain_in_use(tmp_path / "calc.xlsm", platform="win32"):
            raise PermissionError(13, "The process cannot access the file")


def test_elsewhere_a_permission_error_stays_one(tmp_path):
    with pytest.raises(PermissionError) as raised:
        with explain_in_use(tmp_path / "x", platform="darwin"):
            raise PermissionError(13, "denied")
    assert not isinstance(raised.value, FileInUse)


def test_a_refused_replace_leaves_the_old_file(tmp_path, monkeypatch):
    target, part = tmp_path / "calc.xlsm", tmp_path / "calc.xlsm.part"
    target.write_bytes(b"old")
    part.write_bytes(b"new")

    def locked(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(in_use.os, "replace", locked)
    with pytest.raises(FileInUse):
        replace_explained(part, target, platform="win32")
    assert target.read_bytes() == b"old"


def test_a_refused_tree_delete_names_the_file_that_blocked_it(tmp_path, monkeypatch):
    folder = tmp_path / "option1"
    folder.mkdir()
    (folder / "report.docx").write_bytes(b"x")

    def locked(path, onexc=None, **kwargs):
        raise PermissionError(13, "in use", str(folder / "report.docx"))

    monkeypatch.setattr(in_use.shutil, "rmtree", locked)
    with pytest.raises(FileInUse, match=r"close report\.docx"):
        remove_tree_explained(folder, platform="win32")


def test_a_refused_replace_names_the_target_not_the_part(tmp_path, monkeypatch):
    target, part = tmp_path / "calc.xlsm", tmp_path / "calc.xlsm.part"

    def locked(src, dst):
        # What Windows gives: the source in filename, the target in filename2.
        raise PermissionError(13, "in use", str(part), 32, str(target))

    monkeypatch.setattr(in_use.os, "replace", locked)
    with pytest.raises(FileInUse) as raised:
        replace_explained(part, target, platform="win32")
    assert str(raised.value) == (
        "calc.xlsm is open in another program; close calc.xlsm in Excel or Word, then try again")


def test_an_error_with_two_paths_names_the_second(tmp_path):
    with pytest.raises(FileInUse, match=r"^calc\.xlsm is open"):
        with explain_in_use(tmp_path / "x", platform="win32"):
            raise PermissionError(13, "in use", str(tmp_path / "calc.xlsm.part"), 32,
                                  str(tmp_path / "calc.xlsm"))


def test_an_office_owner_file_is_named_as_the_document(tmp_path):
    with pytest.raises(FileInUse) as raised:
        with explain_in_use(tmp_path / "x", platform="win32"):
            raise PermissionError(13, "in use", str(tmp_path / "~$report.docx"))
    assert str(raised.value) == (
        "report.docx is open in another program; close report.docx in Excel or Word, then try again")


def test_a_refused_folder_rename_says_a_file_inside_it_is_open(tmp_path):
    final, old = tmp_path / "case1", tmp_path / ".case1.old"
    with pytest.raises(FileInUse) as raised:
        with explain_in_use(final, folder=True, platform="win32"):
            raise PermissionError(13, "in use", str(final), 32, str(old))
    assert str(raised.value) == (
        "a file inside case1 is open in another program; close it in Excel or Word, then try again")


def test_a_refused_tree_delete_of_the_folder_itself_says_a_file_inside_it_is_open(tmp_path, monkeypatch):
    folder = tmp_path / "option1"
    folder.mkdir()

    def locked(path, onexc=None, **kwargs):
        raise PermissionError(13, "in use", str(folder))

    monkeypatch.setattr(in_use.shutil, "rmtree", locked)
    with pytest.raises(FileInUse) as raised:
        remove_tree_explained(folder, platform="win32")
    assert str(raised.value) == (
        "a file inside option1 is open in another program; close it in Excel or Word, then try again")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows: a file another handle has open")
def test_windows_a_replace_over_an_open_file_names_it_and_leaves_it(tmp_path):
    target, part = tmp_path / "calc.xlsm", tmp_path / "calc.xlsm.part"
    target.write_bytes(b"old")
    part.write_bytes(b"new")
    with open(target, "rb"):
        with pytest.raises(FileInUse) as raised:
            replace_explained(part, target)
    assert str(raised.value) == (
        "calc.xlsm is open in another program; close calc.xlsm in Excel or Word, then try again")
    assert target.read_bytes() == b"old"
