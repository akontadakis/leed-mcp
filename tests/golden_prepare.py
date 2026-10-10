# tests/golden_prepare.py
"""Golden prepared models: every scene master accepts must prepare byte for byte the same.

Ten invented, all-vertical buildings (round numbers throughout, never a
real project), each built here in code. ``prepare`` turns each into the
five outputs a run depends on -- ``scene.rad``, ``materials.rad``,
``blinds/``, ``grids/`` and ``manifest.json`` -- and those were written
once, on the commit before roof glass was supported, into
``tests/fixtures/golden_prepare/<fixture>/``. ``tests/test_golden_prepare.py``
prepares the same buildings on the current code and compares every file
byte for byte: a change that alters any of them would silently invalidate
every cached daylight-coefficient matrix (each fingerprint hashes the whole
manifest).

Nothing in these outputs depends on the machine: ``prepare`` writes only
paths relative to its output folder (``files`` in the manifest), the
project root is never written, every primitive's ``source`` is the fixed
word ``golden`` (validation messages quote it), and the project carries no
warnings of its own.

Generator mode (run once, on the commit the goldens are taken from; it
refuses to overwrite an existing golden folder)::

    PYTHONPATH=. .venv/bin/python tests/golden_prepare.py --write
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from types import MappingProxyType

from leed_daylight.build.prepare import PreparedModel, prepare
from leed_daylight.project.config import (
    Analysis,
    Blinds,
    Glazing,
    Grid,
    Option2Spec,
    Performance,
    Project,
    RadianceParams,
    Reflectance,
    SceneSpec,
    Site,
    Sky,
)
from leed_daylight.project.spaces import Space
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import detect_rooms

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "golden_prepare"
CASE = "base_case"
SOURCE = "golden"

#: The five outputs compared: two files and two folders at the top, plus the manifest.
TOP_LEVEL = frozenset({"scene.rad", "materials.rad", "blinds", "grids", "manifest.json"})


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    prims: tuple[Primitive, ...]
    roles: MappingProxyType
    spaces: tuple[Space, ...]
    glazing: dict  # Glazing keyword arguments that differ from the base


# -- primitives ---------------------------------------------------------


def _material(ptype: str, name: str, reals) -> Primitive:
    return Primitive("void", ptype, name, (), (), tuple(float(v) for v in reals), SOURCE)


def _polygon(modifier: str, name: str, points) -> Primitive:
    flat = tuple(float(v) for p in points for v in p)
    return Primitive(modifier, "polygon", name, (), (), flat, SOURCE)


BASE_MATERIALS = (
    _material("plastic", "wall_paint", (0.5, 0.5, 0.5, 0, 0)),
    _material("plastic", "ceiling_tile", (0.8, 0.8, 0.8, 0, 0)),
    _material("plastic", "floor_carpet", (0.2, 0.2, 0.2, 0, 0)),
    _material("glass", "glass_clear", (0.76, 0.76, 0.76)),
)
BASE_ROLES = {
    "wall_paint": "wall",
    "ceiling_tile": "ceiling",
    "floor_carpet": "floor",
    "glass_clear": "glazing",
}


def _slab(modifier: str, name: str, x0, y0, x1, y1, z, *, up: bool) -> Primitive:
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return _polygon(modifier, name, corners if up else corners[::-1])


def _box(prefix: str, x0, y0, x1, y1, z0, height, *, open_sides=()) -> list[Primitive]:
    """Floor, ceiling and the four walls of one box room; a side in
    ``open_sides`` ("south", "east", "north", "west") gets no wall, so the
    glass there is the only surface in that plane."""
    z1 = z0 + height
    walls = {
        "south": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        "east": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        "north": [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        "west": [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    }
    prims = [
        _slab("floor_carpet", f"floor_{prefix}", x0, y0, x1, y1, z0, up=True),
        _slab("ceiling_tile", f"ceiling_{prefix}", x0, y0, x1, y1, z1, up=False),
    ]
    for side, pts in walls.items():
        if side not in open_sides:
            prims.append(_polygon("wall_paint", f"wall_{prefix}_{side}", pts))
    return prims


def _pane_y(modifier, name, y, x0, x1, z0, z1) -> Primitive:
    """A vertical pane in the plane y = const."""
    return _polygon(modifier, name, [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)])


def _pane_x(modifier, name, x, y0, y1, z0, z1) -> Primitive:
    """A vertical pane in the plane x = const."""
    return _polygon(modifier, name, [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])


def _space(room, area, cx, cy, level_z=0.0, floor="L0") -> Space:
    return Space(
        floor=floor, room=room, space_type="office", regularly_occupied=True,
        area_m2=float(area), centroid_x=float(cx), centroid_y=float(cy), level_z=float(level_z),
        view_preserving_shades=False, merge_into=None, include={CASE: True}, reason="",
    )


# -- the ten fixtures ---------------------------------------------------


def _one_room() -> GoldenCase:
    # A 6 x 4 x 3 m room: two south windows and one east window.
    prims = [
        *BASE_MATERIALS,
        *_box("room_a", 0, 0, 6, 4, 0, 3),
        _pane_y("glass_clear", "south_1", 0, 1, 2, 1, 2),
        _pane_y("glass_clear", "south_2", 0, 4, 5, 1, 2),
        _pane_x("glass_clear", "east_1", 6, 1, 2, 1, 2),
    ]
    return GoldenCase(tuple(prims), MappingProxyType(BASE_ROLES), (_space("room_a", 24, 3, 2),), {})


def _mixed_facades() -> GoldenCase:
    # An 8 x 5 x 3 m room: a south window, a north window, and an east
    # curtain wall of two floor-to-ceiling panes sharing a mullion at y = 2.5.
    prims = [
        *BASE_MATERIALS,
        _material("glass", "glass_cw", (0.8, 0.8, 0.8)),
        *_box("room_b", 0, 0, 8, 5, 0, 3, open_sides=("east",)),
        _pane_y("glass_clear", "south_1", 0, 2, 4, 1, 2),
        _pane_y("glass_clear", "north_1", 5, 3, 5, 1, 2),
        _pane_x("glass_cw", "cw_1", 8, 0, 2.5, 0, 3),
        _pane_x("glass_cw", "cw_2", 8, 2.5, 5, 0, 3),
    ]
    roles = dict(BASE_ROLES, glass_cw="glazing_curtain_wall")
    glazing = {
        "frame_share": MappingProxyType({"default": 0.1, "glazing_curtain_wall": 0.05}),
        "frame_style": "uniform_scale",
    }
    return GoldenCase(tuple(prims), MappingProxyType(roles), (_space("room_b", 40, 4, 2.5),), glazing)


def _set_back_window() -> GoldenCase:
    # A 6 x 5 m room outline traced to the exterior wall face; the south
    # glass is drawn mid-wall, 0.2 m inside that outline.
    prims = [
        *BASE_MATERIALS,
        *_box("room_c", 0, 0, 6, 5, 0, 3, open_sides=("south",)),
        _pane_y("glass_clear", "south_1", 0.2, 2, 4, 1, 2),
    ]
    return GoldenCase(tuple(prims), MappingProxyType(BASE_ROLES), (_space("room_c", 30, 3, 2.5),), {})


def _glass_outside_interior_face() -> GoldenCase:
    # A 6 x 5 m room outline traced to the interior wall face; the south
    # glass sits 0.1 m outside it, in the wall's thickness.
    prims = [
        *BASE_MATERIALS,
        *_box("room_d", 0, 0, 6, 5, 0, 3, open_sides=("south",)),
        _pane_y("glass_clear", "south_1", -0.1, 2, 4, 1, 2),
    ]
    return GoldenCase(tuple(prims), MappingProxyType(BASE_ROLES), (_space("room_d", 30, 3, 2.5),), {})


def _courtyard() -> GoldenCase:
    # A 6 x 6 m plate of four slabs round a 2 x 2 m open courtyard; one
    # window looks south out of the building, one looks north into the courtyard.
    slabs = [("s", 0, 0, 6, 2), ("n", 0, 4, 6, 6), ("w", 0, 2, 2, 4), ("e", 4, 2, 6, 4)]
    prims = [*BASE_MATERIALS]
    for tag, x0, y0, x1, y1 in slabs:
        prims.append(_slab("floor_carpet", f"floor_{tag}", x0, y0, x1, y1, 0, up=True))
        prims.append(_slab("ceiling_tile", f"ceiling_{tag}", x0, y0, x1, y1, 3, up=False))
    prims += [
        _pane_y("glass_clear", "outer_south", 0, 2, 4, 1, 2),
        _pane_y("glass_clear", "court_south", 2, 2.5, 3.5, 1, 2),
    ]
    return GoldenCase(tuple(prims), MappingProxyType(BASE_ROLES), (_space("room_e", 32, 3, 3),), {})


def _two_storeys() -> GoldenCase:
    # Two 6 x 4 m rooms stacked: floors at z = 0 and z = 3.5.
    prims = [
        *BASE_MATERIALS,
        *_box("room_f", 0, 0, 6, 4, 0, 3),
        *_box("room_g", 0, 0, 6, 4, 3.5, 3),
        _pane_y("glass_clear", "lower_south", 0, 2, 4, 1, 2),
        _pane_y("glass_clear", "upper_south", 0, 2, 4, 4.5, 5.5),
        _pane_x("glass_clear", "upper_west", 0, 1, 3, 4.5, 5.5),
    ]
    spaces = (
        _space("room_f", 24, 3, 2),
        _space("room_g", 24, 3, 2, level_z=3.5, floor="L1"),
    )
    return GoldenCase(tuple(prims), MappingProxyType(BASE_ROLES), spaces, {})


def _unused_glazing_material() -> GoldenCase:
    # An exterior glazing material that no surface uses still gets a record.
    base = _one_room()
    prims = (*base.prims[:4], _material("glass", "glass_spare", (0.7, 0.7, 0.7)), *base.prims[4:])
    roles = dict(base.roles, glass_spare="glazing")
    return GoldenCase(prims, MappingProxyType(roles), base.spaces, {})


def _interior_partitions() -> GoldenCase:
    # Two 6 x 4 m rooms 0.2 m apart; the partition wall between them
    # carries interior glass; each room has one south window.
    prims = [
        *BASE_MATERIALS,
        _material("glass", "glass_partition", (0.8, 0.8, 0.8)),
        *_box("room_h", 0, 0, 6, 4, 0, 3, open_sides=("east",)),
        *_box("room_i", 6.2, 0, 12.2, 4, 0, 3, open_sides=("west",)),
        _pane_x("wall_paint", "partition", 6.1, 0, 4, 0, 3),
        _pane_x("glass_partition", "partition_glass", 6.1, 1, 3, 1, 2),
        _pane_y("glass_clear", "south_h", 0, 2, 4, 1, 2),
        _pane_y("glass_clear", "south_i", 0, 8, 10, 1, 2),
    ]
    roles = dict(BASE_ROLES, glass_partition="glazing_interior")
    spaces = (_space("room_h", 24, 3, 2), _space("room_i", 24, 9.2, 2))
    return GoldenCase(tuple(prims), MappingProxyType(roles), spaces, {})


def _tinted_glass() -> GoldenCase:
    # The one-room building with tinted glass: unequal r, g, b
    # transmissivity (kept as tinted, with a warning).
    prims = []
    for p in _one_room().prims:
        if p.name == "glass_clear":
            p = _material("glass", "glass_tint", (0.7, 0.75, 0.65))
        elif p.modifier == "glass_clear":
            p = dataclasses.replace(p, modifier="glass_tint")
        prims.append(p)
    roles = {k: v for k, v in BASE_ROLES.items() if k != "glass_clear"}
    roles["glass_tint"] = "glazing"
    return GoldenCase(tuple(prims), MappingProxyType(roles), (_space("room_a", 24, 3, 2),), {})


def _vt_includes_dirt() -> GoldenCase:
    # The one-room building, its glass VT already net of dirt.
    base = _one_room()
    return GoldenCase(base.prims, base.roles, base.spaces, {"vt_includes_dirt": True})


FIXTURES = {
    "one_room": _one_room,
    "mixed_facades": _mixed_facades,
    "set_back_window": _set_back_window,
    "glass_outside_interior_face": _glass_outside_interior_face,
    "courtyard": _courtyard,
    "two_storeys": _two_storeys,
    "unused_glazing_material": _unused_glazing_material,
    "interior_partitions": _interior_partitions,
    "tinted_glass": _tinted_glass,
    "vt_includes_dirt": _vt_includes_dirt,
}
FIXTURE_NAMES: tuple[str, ...] = tuple(FIXTURES)


def golden_case(name: str) -> GoldenCase:
    return FIXTURES[name]()


def golden_project(root, **glazing_changes) -> Project:
    """The invented project every golden fixture shares; ``glazing_changes``
    replaces single ``Glazing`` fields."""
    glazing = dict(
        dirt_factor=0.9,
        frame_share=MappingProxyType({"default": 0.1}),
        frame_reflectance=0.3,
        frame_style="constant_width",
        reveal_depth_m=0.15,
        reveal_reflectance=0.5,
        vt_includes_dirt=False,
    )
    glazing.update(glazing_changes)
    root = pathlib.Path(root)
    return Project(
        name="Golden Office",
        credit="LEED v4 BD+C, EQ Credit Daylight",
        root=root,
        site=Site(
            latitude_N=40.0, longitude_E=-75.0, meridian_radiance=75.0,
            north_rotation_deg=10.0, north_rotation_source="Invented for the golden fixtures",
        ),
        scene=SceneSpec(units="m", files=()),
        standard="LM-83-12",
        credit_rules_path=root / "credit_rules.yaml",
        weather_file=root / "weather.wea",
        glazing=Glazing(**glazing),
        reflectance=Reflectance(policy="keep_as_modelled", by_role=MappingProxyType({})),
        grid=Grid(spacing_m=0.5, work_plane_m=0.75, min_clearance_m=0.1),
        analysis=Analysis(
            hours="08:00-18:00", daylight_saving="none",
            aggregation="floor_area", moveable_furniture="exclude",
        ),
        blinds=Blinds(0.05, 0.0, 0.5),
        sky=Sky(basis="Reinhart", MF=1),
        radiance=RadianceParams(ambient_draft="-ab 2", ambient_full="-ab 6", direct="-dc 1"),
        performance=Performance(trace_scope="all", linked_zones=MappingProxyType({})),
        option2=Option2Spec(
            equinox_dates=(), clearest_day_window_days=0, times=(),
            sky="CIE clear with sun", lux_min=300.0, lux_max=3000.0,
            blinds="exclude", time_basis="UNVERIFIED",
        ),
        warnings=(),
    )


def prepare_golden(name: str, out_dir) -> PreparedModel:
    """Prepare fixture ``name`` into ``out_dir`` on the code under test."""
    case = golden_case(name)
    prims = list(case.prims)
    rooms = detect_rooms(prims, dict(case.roles))
    project = golden_project(out_dir, **case.glazing)
    return prepare(project, prims, dict(case.roles), rooms, list(case.spaces), CASE, out_dir)


def snapshot(folder) -> dict[str, bytes]:
    """Every file under ``folder``, by relative POSIX path, as bytes."""
    folder = pathlib.Path(folder)
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith(".")  # never a Finder .DS_Store
    }


def write_goldens(golden_dir=GOLDEN_DIR) -> None:
    """Generator mode: prepare every fixture into ``golden_dir``. Refuses,
    before writing anything, if any fixture's folder is already there."""
    golden_dir = pathlib.Path(golden_dir)
    existing = [name for name in FIXTURE_NAMES if (golden_dir / name).exists()]
    if existing:
        raise SystemExit(
            f"{golden_dir}: {', '.join(existing)} already exist; goldens are "
            "written once, on the commit they are taken from, never refreshed"
        )
    for name in FIXTURE_NAMES:
        prepare_golden(name, golden_dir / name)
        print(f"wrote {golden_dir / name}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: PYTHONPATH=. .venv/bin/python tests/golden_prepare.py --write")
    write_goldens()
