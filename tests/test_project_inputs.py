# tests/test_project_inputs.py
"""input_fingerprints / stale_inputs (reports spec 3.2)."""
from __future__ import annotations

import hashlib
import os

import pytest
import yaml

from leed_daylight.project.config import load_project
from leed_daylight.project.inputs import input_fingerprints, materials_digest, stale_inputs

MATERIALS_CSV = "modifier,primitive_type,role,note\n"
SPACES_CSV = "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,reason\n"


def _folder(tmp_path, write_example, mutate=lambda d: d):
    folder = write_example(tmp_path, mutate).parent
    (folder / "materials.csv").write_text(MATERIALS_CSV)
    (folder / "spaces.csv").write_text(SPACES_CSV)
    return folder


def test_every_input_file_is_fingerprinted(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")

    prints = input_fingerprints(project, folder)

    assert list(prints) == sorted(prints)
    assert set(prints) == {
        "credit_rules.yaml", "materials.csv", "project.yaml", "scene.rad",
        "spaces.csv", "weather/example.wea",
    }
    assert prints["spaces.csv"] == hashlib.sha256(SPACES_CSV.encode()).hexdigest()


def test_nothing_is_stale_right_after_fingerprinting(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    assert stale_inputs(recorded, project, folder) == []


def test_one_changed_byte_is_named(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    (folder / "spaces.csv").write_text(SPACES_CSV + " ")

    assert stale_inputs(recorded, project, folder) == ["spaces.csv"]


def test_a_file_known_on_only_one_side_is_named(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = dict(input_fingerprints(project, folder))
    recorded["old_scene.rad"] = "0" * 64

    assert stale_inputs(recorded, project, folder) == ["old_scene.rad"]


def test_a_scene_file_outside_the_folder_is_keyed_by_its_absolute_path(tmp_path, write_example):
    outside = tmp_path / "elsewhere" / "context.rad"
    outside.parent.mkdir()
    outside.write_text("")

    def mutate(d):
        d["scene"]["files"] = ["scene.rad", str(outside)]
        return d

    folder = _folder(tmp_path, write_example, mutate)
    project = load_project(folder / "project.yaml")

    assert str(outside.resolve()) in input_fingerprints(project, folder)


def test_editing_the_report_section_leaves_results_current(tmp_path, write_example):
    """Final review I1 (Ruling FR-I1): report: is presentation only, so
    adding a deviation or prepared_by must not force a Radiance rerun."""
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    with open(folder / "project.yaml", "a") as handle:
        handle.write("report:\n  deviations:\n    - An invented deviation.\n  prepared_by: Example Studio\n")

    assert stale_inputs(recorded, load_project(folder / "project.yaml"), folder) == []


def test_editing_the_atlas_section_leaves_results_current(tmp_path, write_example):
    """Atlas spec 8.1: a drawing under a plan changes no result, so adding
    or moving one must not force a Radiance rerun."""
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    with open(folder / "project.yaml", "a") as handle:
        handle.write(
            "atlas:\n  drawings:\n    - storey: L0\n      image: drawings/level0.png\n      points:\n"
            "        - {pixel: [100, 900], model: [0.0, 0.0]}\n"
            "        - {pixel: [1100, 900], model: [20.0, 0.0]}\n"
        )

    assert stale_inputs(recorded, load_project(folder / "project.yaml"), folder) == []


def test_changing_any_other_project_key_makes_results_stale(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["project"]["name"] = "Example Tower Two"
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    assert stale_inputs(recorded, load_project(folder / "project.yaml"), folder) == ["project.yaml"]


# -- splits.csv (merge-split spec 8) ----------------------------------------

SPLITS_CSV = 'level_z,points,note\n0.0,"2.0 0.0; 2.0 4.0",\n'


def test_splits_csv_is_fingerprinted_only_when_it_exists(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    assert "splits.csv" not in input_fingerprints(project, folder)

    (folder / "splits.csv").write_text(SPLITS_CSV)
    prints = input_fingerprints(project, folder)

    assert list(prints) == sorted(prints)
    assert prints["splits.csv"] == hashlib.sha256(SPLITS_CSV.encode()).hexdigest()


def test_a_new_splits_csv_makes_results_recorded_without_it_stale(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    (folder / "splits.csv").write_text(SPLITS_CSV)

    assert stale_inputs(recorded, project, folder) == ["splits.csv"]


def test_editing_splits_csv_makes_results_stale(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    (folder / "splits.csv").write_text(SPLITS_CSV)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    (folder / "splits.csv").write_text(SPLITS_CSV.replace("2.0 4.0", "2.0 3.0"))

    assert stale_inputs(recorded, project, folder) == ["splits.csv"]


def test_removing_splits_csv_makes_results_stale(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    (folder / "splits.csv").write_text(SPLITS_CSV)
    project = load_project(folder / "project.yaml")
    recorded = input_fingerprints(project, folder)

    (folder / "splits.csv").unlink()

    assert stale_inputs(recorded, project, folder) == ["splits.csv"]


def test_no_splits_csv_leaves_the_fingerprint_unchanged(tmp_path, write_example):
    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")

    prints = input_fingerprints(project, folder)

    # The same six keys as before splits.csv existed, so results recorded
    # before this change stay current.
    assert list(prints) == [
        "credit_rules.yaml", "materials.csv", "project.yaml", "scene.rad",
        "spaces.csv", "weather/example.wea",
    ]
    assert stale_inputs(prints, project, folder) == []


# -- an input is hashed as the input reader reads it (bounded, plain files only) --


def test_both_hashes_of_a_normal_file_are_its_plain_digest(tmp_path):
    from leed_daylight.project.inputs import _sha256_input, sha256_file

    data = bytes(range(256)) * 9000  # more than one chunk
    path = tmp_path / "data.bin"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
    assert _sha256_input(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_an_input_hash_refuses_a_fifo_without_blocking(tmp_path):
    import os

    import pytest

    from leed_daylight.project.inputs import _sha256_input
    from leed_daylight.scene.input_file import InputFileError

    fifo = tmp_path / "materials.csv"
    os.mkfifo(fifo)

    from test_input_limits import BLOCKED_S, _within

    box = _within(BLOCKED_S, lambda: _sha256_input(fifo))
    assert isinstance(box.get("error"), InputFileError)
    assert str(box["error"]) == f"{fifo}: is not a plain file; it is not read"


def test_an_input_hash_refuses_a_file_over_the_cap(tmp_path, monkeypatch):
    import pytest

    from leed_daylight.project.inputs import _sha256_input
    from leed_daylight.scene import input_file

    monkeypatch.setattr(input_file, "INPUT_FILE_MAX", 10)
    path = tmp_path / "spaces.csv"
    path.write_bytes(b"x" * 11)

    with pytest.raises(input_file.InputFileError, match="is larger than 10 bytes"):
        _sha256_input(path)
    path.write_bytes(b"x" * 10)
    assert _sha256_input(path) == hashlib.sha256(b"x" * 10).hexdigest()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_a_fifo_input_is_refused_by_the_fingerprints(tmp_path, write_example):
    import os

    import pytest

    from leed_daylight.scene.input_file import InputFileError

    folder = _folder(tmp_path, write_example)
    project = load_project(folder / "project.yaml")
    (folder / "spaces.csv").unlink()
    os.mkfifo(folder / "spaces.csv")

    from test_input_limits import BLOCKED_S, _within

    box = _within(BLOCKED_S, lambda: input_fingerprints(project, folder))
    assert isinstance(box.get("error"), InputFileError)
    assert "spaces.csv: is not a plain file" in str(box["error"])


@pytest.mark.parametrize("before, after", [
    ("modifier,primitive_type,role,note\nglass_a,glass,glazing,\n",
     "modifier,primitive_type,role,note,vlt\nglass_a,glass,glazing,,62\n"),
    ("modifier,primitive_type,role,note,view\nglass_a,glass,glazing,,clear\n",
     "modifier,primitive_type,role,note,view,vlt\nglass_a,glass,glazing,,clear,62\n"),
])
def test_adding_vlt_keeps_the_materials_fingerprint(tmp_path, before, after):
    (tmp_path / "a.csv").write_text(before, encoding="utf-8")
    (tmp_path / "b.csv").write_text(after, encoding="utf-8")
    assert materials_digest(tmp_path / "a.csv") == materials_digest(tmp_path / "b.csv")


def test_a_role_change_still_changes_the_fingerprint(tmp_path):
    (tmp_path / "a.csv").write_text("modifier,primitive_type,role,note\nglass_a,glass,glazing,\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("modifier,primitive_type,role,note,vlt\nglass_a,glass,wall,,\n", encoding="utf-8")
    assert materials_digest(tmp_path / "a.csv") != materials_digest(tmp_path / "b.csv")


def test_a_hand_added_vlt_keeps_a_crlf_bom_file_quoted_note_and_empty_view(tmp_path):
    before = ('﻿modifier,primitive_type,role,note,view\r\n'
              'glass_a,glass,glazing,"a note, with ""quotes""",\r\n\r\nwall_a,plastic,wall,,\r\n')
    after = ('﻿modifier,primitive_type,role,note,view,vlt\r\n'
             'glass_a,glass,glazing,"a note, with ""quotes""",,62\r\n\r\nwall_a,plastic,wall,,,\r\n')
    (tmp_path / "a.csv").write_bytes(before.encode("utf-8"))
    (tmp_path / "b.csv").write_bytes(after.encode("utf-8"))
    assert materials_digest(tmp_path / "a.csv") == materials_digest(tmp_path / "b.csv")


def _atlas_folder(tmp_path, write_example):
    def mutate(d):
        d["atlas"] = {"drawings": [{"storey": "L0", "image": "level0.png", "points": [
            {"pixel": [0, 0], "model": [0, 0]}, {"pixel": [100, 0], "model": [10, 0]}]}]}
        return d

    folder = _folder(tmp_path, write_example, mutate)
    (folder / "level0.png").write_bytes(b"\x89PNG one")
    return folder


def test_the_presentation_digest_follows_the_report_and_atlas_sections_and_the_drawings(tmp_path, write_example):
    from leed_daylight.project.inputs import presentation_digest

    folder = _atlas_folder(tmp_path, write_example)
    before = presentation_digest(load_project(folder / "project.yaml"), folder)
    assert before == presentation_digest(load_project(folder / "project.yaml"), folder)

    (folder / "level0.png").write_bytes(b"\x89PNG two")
    after_drawing = presentation_digest(load_project(folder / "project.yaml"), folder)
    assert after_drawing != before

    path = folder / "project.yaml"
    path.write_text(path.read_text() + "report:\n  prepared_by: Someone\n")
    assert presentation_digest(load_project(path), folder) != after_drawing


def test_the_presentation_digest_ignores_the_rest_of_project_yaml(tmp_path, write_example):
    from leed_daylight.project.inputs import presentation_digest

    folder = _folder(tmp_path, write_example)
    before = presentation_digest(load_project(folder / "project.yaml"), folder)
    path = folder / "project.yaml"
    path.write_text(path.read_text().replace("spacing_m: 0.60", "spacing_m: 0.70"))

    assert presentation_digest(load_project(path), folder) == before
