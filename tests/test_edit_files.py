# tests/test_wizard_files.py
"""Safe saving, backups, change detection, the path rule and the recent
projects file of the wizard (wizard spec 3, 7.1, 10).

Every folder is built fresh in ``tmp_path``; no test touches the real
``~/.leed-daylight/`` (``tests/conftest.py`` points the recent file into a
temporary folder).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib

import pytest

from conftest import symlink_or_skip
from leed_daylight.edit import files
from leed_daylight.edit.files import (
    KEEP_BACKUPS,
    ChangedOutside,
    PathOutside,
    RecentEntry,
    Served,
    add_recent,
    backup,
    read_recent,
    remove_recent,
    resolve_inside,
    save_files,
    sha256_or_none,
)

NOON = datetime.datetime(2026, 1, 1, 12, 0, 0)
OLD = b"modifier,primitive_type,role,note\nwall_paint,plastic,wall,\n"
NEW = b"modifier,primitive_type,role,note\nwall_paint,plastic,ceiling,\n"


def _nothing(parts):
    pass


def _folder(tmp_path) -> pathlib.Path:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "materials.csv").write_bytes(OLD)
    return folder


def _backups(folder: pathlib.Path, name: str = "materials.csv") -> list[str]:
    where = folder.joinpath(*files.BACKUP_DIR)
    return sorted(p.name for p in where.glob(f"{name}.*")) if where.is_dir() else []


# --- save_files (11.4a) ---------------------------------------------------

def test_a_save_writes_a_part_file_then_replaces(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")
    seen = {}

    def check(parts):
        seen.update(parts)
        assert parts["materials.csv"].read_bytes() == NEW
        assert (folder / "materials.csv").read_bytes() == OLD

    written = save_files(folder, {"materials.csv": NEW}, check=check, served=served, now=NOON)

    assert written == ("materials.csv",)
    assert seen == {"materials.csv": (folder / "materials.csv.part").resolve()}
    assert (folder / "materials.csv").read_bytes() == NEW
    assert not list(folder.glob("*.part"))


def test_a_failed_check_leaves_the_target_and_no_part(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")
    served.remember(folder, "spaces.csv")

    def check(parts):
        raise ValueError("materials.csv: an invented refusal")

    with pytest.raises(ValueError, match="an invented refusal"):
        save_files(folder, {"materials.csv": NEW, "spaces.csv": b"floor\n"}, check=check,
                   served=served, now=NOON)

    assert (folder / "materials.csv").read_bytes() == OLD
    assert not (folder / "spaces.csv").exists()
    assert not list(folder.glob("*.part"))
    assert not (folder / "_run").exists()


def test_a_backup_is_made_of_the_old_file(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")
    served.remember(folder, "spaces.csv")

    save_files(folder, {"materials.csv": NEW, "spaces.csv": b"floor\n"}, check=_nothing,
               served=served, now=NOON)

    kept = folder / "_run" / "backup" / "materials.csv.20260101-120000"
    assert kept.read_bytes() == OLD
    assert _backups(folder) == ["materials.csv.20260101-120000"]
    assert _backups(folder, "spaces.csv") == []


def test_only_twenty_backups_are_kept(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")

    for minute in range(25):
        data = f"{minute}\n".encode()
        save_files(folder, {"materials.csv": data}, check=_nothing, served=served,
                   now=NOON + datetime.timedelta(minutes=minute))

    kept = _backups(folder)
    assert len(kept) == KEEP_BACKUPS == 20
    assert kept[0] == "materials.csv.20260101-120500"
    assert kept[-1] == "materials.csv.20260101-122400"
    assert (folder / "_run" / "backup" / kept[0]).read_bytes() == b"4\n"


def test_two_saves_in_one_second_get_two_backups(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")

    save_files(folder, {"materials.csv": NEW}, check=_nothing, served=served, now=NOON)
    save_files(folder, {"materials.csv": b"third\n"}, check=_nothing, served=served, now=NOON)
    save_files(folder, {"materials.csv": b"fourth\n"}, check=_nothing, served=served, now=NOON)

    assert _backups(folder) == [
        "materials.csv.20260101-120000", "materials.csv.20260101-120000-2", "materials.csv.20260101-120000-3",
    ]
    where = folder / "_run" / "backup"
    assert (where / "materials.csv.20260101-120000-2").read_bytes() == NEW


def test_the_newest_twenty_are_kept_across_clashes(tmp_path):
    folder = _folder(tmp_path)
    for _ in range(22):
        backup(folder, "materials.csv", now=NOON)

    kept = _backups(folder)
    assert len(kept) == 20
    assert "materials.csv.20260101-120000" not in kept
    assert "materials.csv.20260101-120000-2" not in kept
    assert "materials.csv.20260101-120000-22" in kept


def test_a_file_changed_outside_the_wizard_is_refused(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")
    (folder / "materials.csv").write_bytes(b"edited by hand\n")

    with pytest.raises(ChangedOutside) as caught:
        save_files(folder, {"materials.csv": NEW}, check=_nothing, served=served, now=NOON)

    assert str(caught.value) == "materials.csv was changed outside the wizard; reload the step"
    assert (folder / "materials.csv").read_bytes() == b"edited by hand\n"
    assert not list(folder.glob("*.part"))
    assert not (folder / "_run").exists()


def test_a_file_never_served_is_refused(tmp_path):
    folder = _folder(tmp_path)

    with pytest.raises(ChangedOutside, match=r"^materials\.csv was changed outside the wizard; reload the step$"):
        save_files(folder, {"materials.csv": NEW}, check=_nothing, served=Served(), now=NOON)

    assert (folder / "materials.csv").read_bytes() == OLD


def test_a_file_missing_when_served_may_be_made_unless_it_appeared_since(tmp_path):
    folder = _folder(tmp_path)
    served = Served()

    assert served.remember(folder, "splits.csv") is None
    save_files(folder, {"splits.csv": b"level_z,points,note\n"}, check=_nothing, served=served, now=NOON)
    assert (folder / "splits.csv").read_bytes() == b"level_z,points,note\n"

    served.remember(folder, "spaces.csv")
    (folder / "spaces.csv").write_bytes(b"made by hand\n")
    with pytest.raises(ChangedOutside):
        save_files(folder, {"spaces.csv": b"floor\n"}, check=_nothing, served=served, now=NOON)


def test_a_save_remembers_what_it_wrote(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "materials.csv")

    save_files(folder, {"materials.csv": NEW}, check=_nothing, served=served, now=NOON)
    save_files(folder, {"materials.csv": OLD}, check=_nothing, served=served, now=NOON)

    served.clear()
    with pytest.raises(ChangedOutside):
        served.check(folder, "materials.csv")


def test_a_name_with_a_sub_folder_is_saved_and_backed_up_under_it(tmp_path):
    folder = _folder(tmp_path)
    served = Served()
    served.remember(folder, "drawings/L0.png")

    save_files(folder, {"drawings/L0.png": b"first"}, check=_nothing, served=served, now=NOON)
    save_files(folder, {"drawings/L0.png": b"second"}, check=_nothing, served=served, now=NOON)

    assert (folder / "drawings" / "L0.png").read_bytes() == b"second"
    kept = folder / "_run" / "backup" / "drawings" / "L0.png.20260101-120000"
    assert kept.read_bytes() == b"first"


def test_a_save_name_follows_the_path_rule(tmp_path):
    folder = _folder(tmp_path)

    with pytest.raises(PathOutside):
        save_files(folder, {"../outside.csv": b"x"}, check=_nothing, served=Served(), now=NOON)

    assert not (tmp_path / "outside.csv").exists()


def test_sha256_or_none(tmp_path):
    folder = _folder(tmp_path)

    assert sha256_or_none(folder / "materials.csv") == hashlib.sha256(OLD).hexdigest()
    assert sha256_or_none(folder / "missing.csv") is None
    assert sha256_or_none(folder) is None


# --- The path rule (11.9) ------------------------------------------------

@pytest.mark.parametrize("rel", ["../x.csv", "model/../../x.csv", "model\\..\\..\\x.csv", ".."])
def test_a_path_with_dot_dot_is_refused(tmp_path, rel):
    folder = _folder(tmp_path)

    with pytest.raises(PathOutside, match="may not use '..'"):
        resolve_inside(folder, rel)


@pytest.mark.parametrize("rel", ["/etc/hosts", "\\temp\\x.csv", "C:\\temp\\x.csv", "C:x.csv", "\\\\server\\share\\x.csv"])
def test_an_absolute_drive_or_unc_path_is_refused(tmp_path, rel):
    folder = _folder(tmp_path)

    with pytest.raises(PathOutside, match="not an absolute one"):
        resolve_inside(folder, rel)


@pytest.mark.parametrize("rel", ["", "   "])
def test_an_empty_path_is_refused(tmp_path, rel):
    with pytest.raises(PathOutside, match="give a path inside the project folder"):
        resolve_inside(_folder(tmp_path), rel)


def test_a_symlink_leading_out_is_refused(tmp_path):
    folder = _folder(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "secret.txt").write_text("invented", newline="\n")
    symlink_or_skip(folder / "model", tmp_path / "elsewhere")
    symlink_or_skip(folder / "link.txt", tmp_path / "elsewhere" / "secret.txt")

    with pytest.raises(PathOutside, match="leads outside the project folder"):
        resolve_inside(folder, "model/secret.txt")
    with pytest.raises(PathOutside, match="leads outside the project folder"):
        resolve_inside(folder, "link.txt")


def test_a_symlink_inside_the_folder_is_allowed(tmp_path):
    folder = _folder(tmp_path)
    (folder / "model").mkdir()
    (folder / "model" / "scene.rad").write_text("", newline="\n")
    symlink_or_skip(folder / "scene_link.rad", folder / "model" / "scene.rad")

    assert resolve_inside(folder, "scene_link.rad") == (folder / "model" / "scene.rad").resolve()
    assert resolve_inside(folder, "model/scene.rad") == (folder / "model" / "scene.rad").resolve()
    assert resolve_inside(folder, "model/new.rad") == (folder / "model" / "new.rad").resolve()


# --- Recent projects -------------------------------------------------------

def test_recent_holds_at_most_ten_newest_first_without_repeats(tmp_path):
    recent = tmp_path / "home" / "recent.json"
    folders = []
    for n in range(12):
        folder = tmp_path / f"project{n}"
        folder.mkdir()
        folders.append(folder)
        add_recent(recent, folder)
    add_recent(recent, folders[5])

    entries = read_recent(recent)

    assert [e.folder for e in entries] == [str(folders[n]) for n in (5, 11, 10, 9, 8, 7, 6, 4, 3, 2)]
    assert all(e.exists for e in entries)
    assert json.loads(recent.read_text()) == [e.folder for e in entries]


def test_a_missing_folder_is_shown_as_missing_and_can_be_removed(tmp_path):
    recent = tmp_path / "recent.json"
    kept = tmp_path / "kept"
    kept.mkdir()
    gone = tmp_path / "gone"
    add_recent(recent, kept)
    add_recent(recent, gone)

    assert read_recent(recent) == [RecentEntry(str(gone), False), RecentEntry(str(kept), True)]

    remove_recent(recent, str(gone))
    assert read_recent(recent) == [RecentEntry(str(kept), True)]


def test_a_relative_folder_is_stored_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recent = tmp_path / "recent.json"

    add_recent(recent, pathlib.Path("project"))

    assert json.loads(recent.read_text()) == [str(tmp_path / "project")]


@pytest.mark.parametrize("text", ["not json", "{\"folders\": []}", "[1, \"relative/path\"]", ""])
def test_a_malformed_recent_file_reads_as_empty(tmp_path, text):
    recent = tmp_path / "recent.json"
    recent.write_text(text, newline="\n")

    assert read_recent(recent) == []


def test_a_missing_recent_file_reads_as_empty(tmp_path):
    assert read_recent(tmp_path / "none" / "recent.json") == []


def test_no_test_touches_the_real_recent_file(tmp_path_factory):
    path = files.default_recent_file()

    assert path.name == "recent.json"
    # In the test run's own temporary folder, which on Windows lies inside
    # the home folder (AppData\Local\Temp), so "not under home" is no test.
    assert tmp_path_factory.getbasetemp().resolve() in path.resolve().parents
    assert path.resolve() != (pathlib.Path.home() / ".leed-daylight" / "recent.json").resolve()
    assert path == files.default_recent_file()
    assert os.path.isabs(path)
