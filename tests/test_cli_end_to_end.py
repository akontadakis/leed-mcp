# tests/test_cli_end_to_end.py
"""From an empty folder to a prepared model, through ``leed-daylight`` only
(CLI spec 8, end to end).

``init`` writes the folder; the test then does what a person does: fills
every FILL_ME (here from the invented ``tests/fixtures/project_example.yaml``,
pointed at this folder's own files), adds an invented one-room scene and
weather file, runs ``scan`` and ``materials``, fills the template's roles,
writes spaces.csv, and runs ``prepare``. No Radiance binary is run.
"""
from __future__ import annotations

import csv
import json
import pathlib

import yaml

from leed_daylight import cli
from leed_daylight.constants import FILL_ME

from conftest import wea_text

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SCENE_RAD = """\
void plastic wall_paint
0
0
5  0.6 0.6 0.6  0 0

void plastic ceiling_tile
0
0
5  0.8 0.8 0.8  0 0

void plastic floor_carpet
0
0
5  0.3 0.3 0.3  0 0

void glass glass_clear
0
0
3  0.6 0.6 0.6

floor_carpet polygon floor_room_north
0
0
12  0 0 0   6 0 0   6 4 0   0 4 0

ceiling_tile polygon ceiling_room_north
0
0
12  0 0 3   0 4 3   6 4 3   6 0 3

wall_paint polygon wall_room_north_south
0
0
12  0 0 0   6 0 0   6 0 3   0 0 3

glass_clear polygon window_ext
0
0
12  2 0 1   4 0 1   4 0 2   2 0 2
"""

ROLES = {"wall_paint": "wall", "ceiling_tile": "ceiling", "floor_carpet": "floor", "glass_clear": "glazing"}

SPACES_CSV = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,
"""


def _fill(raw, example, path=""):
    for key, value in raw.items():
        here = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            _fill(value, example[key], here)
        elif value == FILL_ME:
            raw[key] = example[key]


def test_empty_folder_to_prepared_model(tmp_path, capsys):
    folder = tmp_path / "office_a"
    assert cli.main(["init", str(folder)]) == 0

    raw = yaml.safe_load((folder / "project.yaml").read_text())
    _fill(raw, yaml.safe_load((FIXTURES / "project_example.yaml").read_text()))
    raw["scene"]["files"] = ["model/scene.rad"]
    raw["weather"]["file"] = "weather/site.wea"
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    (folder / "model" / "scene.rad").write_text(SCENE_RAD, newline="\n")
    (folder / "weather" / "site.wea").write_text(wea_text(), newline="\n")

    assert cli.main(["scan", str(folder)]) == 0
    assert cli.main(["materials", str(folder)]) == 0
    with (folder / "_run" / "materials_template.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["role"] = ROLES[row["modifier"]]
    with (folder / "materials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["modifier", "primitive_type", "role", "note"])
        writer.writeheader()
        writer.writerows(rows)
    (folder / "spaces.csv").write_text(SPACES_CSV, newline="\n")

    assert cli.main(["scan", str(folder)]) == 0
    assert cli.main(["prepare", str(folder)]) == 0
    manifest = json.loads((folder / "_run" / "manifest.json").read_text())
    assert manifest["case"] == "base"
    assert manifest["totals"]["group_count"] == 1
    out = capsys.readouterr().out
    assert "storey 0: level 0.00 m, 1 room(s), 24.00 m2" in out
    assert "Room North: 24.00 m2" in out
