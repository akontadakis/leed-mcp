# tests/merge_split_folder.py
"""Fictional project folders with a split room or a merged space, for the
merge-split driver and run tests (merge-split plan Tasks 8 and 9).

Both start from ``tests/option2_folder.py``'s invented project: one
6 m x 4 m x 3 m room, "Room North", with a 2 m wide south window between
x = 2 and 4 m.

- The **split** project adds ``splits.csv`` with one line along x = 4.5 m,
  run north: half ``a`` (west, 18 m2) is the space "Room West" and half
  ``b`` (east, 6 m2, no window) the space "Room Strip".
- The **merge** project adds a second room east of a 1 m gap, "Room East"
  (x 7 to 11 m, 16 m2, a south window between x = 8 and 10 m), and a small
  pane in the gap, 0.5 m from both rooms; Room East is a piece of Room North
  (``merge_into``), so the gap pane ties between the two pieces of one space
  and goes to the first.
- The **combined** project (the opt-in Radiance run) has both: the merge
  project's scene, Room North split along x = 4.5 m into Room West (a split
  space) and Room Strip, and Room Strip a piece of Room East (a split+merged
  space); the gap pane ties between Room East and Room Strip.

Every number is invented and round.
"""
from __future__ import annotations

import pathlib

from leed_daylight.build.prepare import prepare
from leed_daylight.project.config import load_project
from leed_daylight.project.layout import scene_rooms
from leed_daylight.project.materials import load_materials
from leed_daylight.project.spaces import load_spaces
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.validate import validate_scene
from option2_folder import SCENE_RAD, build_option2_project

CASE = "base_case"
VIEWS = {"eye_height_m": 1.2, "eye_height_source": "Fictional example: seated eye height"}

HEADER = (
    "floor,room,space_type,regularly_occupied,area_m2,centroid_x,centroid_y,level_z,"
    "view_preserving_shades,merge_into,include_base_case,views_base_case,reason\n"
)

#: One line along x = 4.5 m, run north, past both long walls of Room North.
SPLITS_CSV = 'level_z,points,note\n0.0,"4.5 -1; 4.5 5",invented west edge\n'

SPLIT_SPACES_CSV = HEADER + (
    "L0,Room West,office,yes,18,2.25,2.0,0.0,no,,1,1,\n"
    "L0,Room Strip,office,yes,6,5.25,2.0,0.0,no,,1,1,\n"
)

EAST_ROOM_RAD = """
floor_carpet polygon floor_room_east
0
0
12  7 0 0   11 0 0   11 4 0   7 4 0

ceiling_tile polygon ceiling_room_east
0
0
12  7 0 3   7 4 3   11 4 3   11 0 3

wall_paint polygon wall_room_east_south
0
0
12  7 0 0   11 0 0   11 0 3   7 0 3

glass_clear polygon window_east
0
0
12  8 0 1   10 0 1   10 0 2   8 0 2
"""

#: A south pane in the 1 m gap between the two rooms: its centre is 0.5 m
#: from each, so it ties between them.
GAP_PANE_RAD = """
glass_clear polygon window_gap
0
0
12  6.25 0 1   6.75 0 1   6.75 0 2   6.25 0 2
"""

MERGE_SCENE_RAD = SCENE_RAD + EAST_ROOM_RAD + GAP_PANE_RAD

MERGE_SPACES_CSV = HEADER + (
    "L0,Room North,office,yes,24,3.0,2.0,0.0,no,,1,1,\n"
    "L0,Room East,office,yes,16,9.0,2.0,0.0,no,Room North,1,1,\n"
)

COMBINED_SPACES_CSV = HEADER + (
    "L0,Room West,office,yes,18,2.25,2.0,0.0,no,,1,1,\n"
    "L0,Room East,office,yes,16,9.0,2.0,0.0,no,,1,1,\n"
    "L0,Room Strip,office,yes,6,5.25,2.0,0.0,no,Room East,1,1,\n"
)


def _mutate(d):
    d["views"] = dict(VIEWS)


def build_split_project(tmp_path, write_example, write_wea) -> pathlib.Path:
    """Room North cut along x = 4.5 m into Room West and Room Strip."""
    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=SPLIT_SPACES_CSV,
                                   mutate=_mutate)
    (folder / "splits.csv").write_text(SPLITS_CSV)
    return folder


def build_merge_project(tmp_path, write_example, write_wea) -> pathlib.Path:
    """Room North and Room East, one space of two pieces, with the gap pane."""
    return build_option2_project(tmp_path, write_example, write_wea, spaces_csv=MERGE_SPACES_CSV,
                                 mutate=_mutate, scene_rad=MERGE_SCENE_RAD)


def build_combined_project(tmp_path, write_example, write_wea, *, mutate=None) -> pathlib.Path:
    """A split space (Room West) and a split+merged one (Room East, with Room
    Strip); ``mutate`` edits project.yaml after the views section is added."""
    def both(d):
        _mutate(d)
        if mutate is not None:
            mutate(d)

    folder = build_option2_project(tmp_path, write_example, write_wea, spaces_csv=COMBINED_SPACES_CSV,
                                   mutate=both, scene_rad=MERGE_SCENE_RAD)
    (folder / "splits.csv").write_text(SPLITS_CSV)
    return folder


def prepared_model(folder: pathlib.Path, case: str = CASE):
    """The model the run commands prepare, split lines applied, into ``<folder>/_run``."""
    project = load_project(folder / "project.yaml")
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    rooms = scene_rooms(project, prims, roles)
    spaces = load_spaces(folder / "spaces.csv")
    return prepare(project, prims, roles, rooms, spaces, case, folder / "_run")
