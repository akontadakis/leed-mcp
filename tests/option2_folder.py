# tests/option2_folder.py
"""A small fictional project folder for the Option 2 tests.

One 6 m x 4 m x 3 m room with a south window (the same invented room as
tests/test_run_command.py), on the example project's invented site
(42 N, 80 W, meridian 75 W), with a weather file whose only direct sun
falls on two invented clear days. Every number is invented and round.
"""
from __future__ import annotations

import calendar
import pathlib

from leed_daylight.build.prepare import prepare
from leed_daylight.project.config import load_project
from leed_daylight.project.materials import load_materials
from leed_daylight.project.rules import load_rules
from leed_daylight.project.spaces import load_spaces
from leed_daylight.project.weather import read_wea
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms
from leed_daylight.scene.validate import validate_scene

ROOM = "Room North"
MARCH_DAY = (3, 28)
SEPTEMBER_DAY = (9, 14)
CLEAR_DNI = 700.0
DAY_DHI = 100.0

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

MATERIALS_CSV = """\
modifier,primitive_type,role,note
wall_paint,plastic,wall,
ceiling_tile,plastic,ceiling,
floor_carpet,plastic,floor,
glass_clear,glass,glazing,
"""

SPACES_CSV = """\
floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,view_preserving_shades,merge_into,include_base_case,reason
L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,
"""

#: tests/conftest.py's MINIMAL_CREDIT_RULES with LEED v4.1 Table 2 bands.
CREDIT_RULES = """\
version: "Example rules v0"
max_points: 3
option1:
  sda:
    lux: 300
    fraction: 0.50
    bands:
      - {min_pct: 40.0, points: 1}
  ase:
    lux: 1000
    hours: 250
    per_space_narrative_trigger_pct: 10.0
option2:
  bands:
    - {min_pct: 55.0, points: 1}
    - {min_pct: 75.0, points: 2}
    - {min_pct: 90.0, points: 3}
  lux_min: 300
  lux_max: 3000
glare_control_required: true
"""


def sunny_wea_lines(clear_days=None, *, dhi_hours=range(6, 18)):
    """8760 .wea data lines: CLEAR_DNI from 07:00 to 17:00 on each clear
    day (default MARCH_DAY and SEPTEMBER_DAY), none on any other day;
    DAY_DHI in the hours `dhi_hours` of every day."""
    clear = {MARCH_DAY: CLEAR_DNI, SEPTEMBER_DAY: CLEAR_DNI} if clear_days is None else dict(clear_days)
    dhi_hours = set(dhi_hours)
    lines = []
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2023, month)[1] + 1):
            for hour in range(24):
                dni = clear.get((month, day), 0.0) if 7 <= hour <= 16 else 0.0
                dhi = DAY_DHI if hour in dhi_hours else 0.0
                lines.append(f"{month} {day} {hour + 0.5} {dni} {dhi}")
    return lines


def build_option2_project(tmp_path, write_example, write_wea, *, spaces_csv=SPACES_CSV,
                          wea_lines=None, mutate=None, scene_rad=SCENE_RAD) -> pathlib.Path:
    """A fresh project folder ready for Option 2."""
    def _mutate(d):
        if mutate is not None:
            mutate(d)
        return d

    folder = write_example(tmp_path, mutate=_mutate).parent
    (folder / "scene.rad").write_text(scene_rad)
    (folder / "materials.csv").write_text(MATERIALS_CSV)
    (folder / "spaces.csv").write_text(spaces_csv)
    (folder / "credit_rules.yaml").write_text(CREDIT_RULES)
    write_wea(
        folder / "weather" / "example.wea", latitude=42.0, longitude_W=80.0,
        hours=wea_lines if wea_lines is not None else sunny_wea_lines(),
    )
    return folder


def load_chain(folder: pathlib.Path, case: str = "base_case"):
    """project, prepared, spaces, rules, weather -- as the option2 command builds them."""
    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    rooms = detect_rooms(prims, roles)
    spaces = load_spaces(folder / "spaces.csv")
    prepared = prepare(project, prims, roles, rooms, spaces, case, folder / "_run")
    rules = load_rules(project.credit_rules_path)
    weather = read_wea(project.weather_file)
    return project, prepared, spaces, rules, weather
