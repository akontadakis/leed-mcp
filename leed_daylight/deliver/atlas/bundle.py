# leed_daylight/deliver/atlas/bundle.py
"""The data behind the atlas page and the Blender file (atlas spec 5, 6, 7).

``build_bundle`` reads the prepared model -- ``scene_file`` and every blind
file, with ``read_scene([...], "m")``: the prepared files are already in
metres, whatever the project's scene units -- and gives each surface
exactly one kind (``classify``) and a storey (``storey_of_z``). The
model-check points are the prepared grids; the points of each result layer
are read, with their values, from that option's ``points.csv``, and take
their storey from their room (``CaseData.storey_of``), never from the
``storey`` column. A point whose room is not in spaces.csv is left out
(``load_case`` has already warned). The room labels are the space rows only
(merge-split spec 7): a space of several pieces is labelled once, where the
plans label it (``StoreyPlan.centroids``: on its largest piece). ``write_npz``
writes the bundle as plain arrays for the Blender script, which cannot
import this package.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from ...build import prepare as _prepare
from ...build.prepare import PreparedModel
from ...project.materials import EXTERIOR_GLAZING_ROLES
from ...project.spaces import space_rows
from ...scene.primitives import Primitive
from ...scene.reader import read_scene
from ..data import COMMANDS, CaseData, DeliverError
from ..figures.spec import NEUTRAL_RGB
from .colours import KIND_RGBA, KINDS
from .layers import Layer, layer_rgb, result_layers

#: The storey of a context or ground surface.
SITE = -1
#: The storey of a surface below the lowest level.
BELOW = -2
#: A surface this close under a level counts as at the level (a millimetre
#: scene read in metres can land a hair below it).
_LEVEL_TOL_M = 1e-6
_SITE_KINDS = frozenset({"context", "ground"})
#: The parts ``build.prepare`` adds, by their fixed modifier names.
_MODIFIER_KINDS = {
    _prepare._FRAME_MATERIAL_NAME: "frame",
    _prepare._REVEAL_MATERIAL_NAME: "reveal",
    _prepare._WELL_MATERIAL_NAME: "skylight_well",
}
_ROLE_KINDS = {
    "glazing": "glass_window", "glazing_curtain_wall": "glass_curtain_wall",
    "glazing_interior": "glass_interior", "glazing_diffusing": "glass_diffusing",
    "wall": "wall", "floor": "floor", "ceiling": "ceiling", "fixed_shading": "fixed_shading",
    "furniture": "furniture", "context": "context", "ground": "ground",
}
_ROOF_TILT_CLASSES = frozenset({"skylight", "sloped"})


@dataclasses.dataclass(frozen=True, eq=False)
class SurfaceSet:
    """The polygons of one kind: ``counts[p]`` vertices each, one after
    another in ``verts``, and the storey of each polygon."""
    verts: np.ndarray  # (M, 3) float64, metres
    counts: np.ndarray  # (P,) int32
    storey: np.ndarray  # (P,) int16


@dataclasses.dataclass(frozen=True, eq=False)
class PointSet:
    xyz: np.ndarray  # (N, 3) float64, metres
    room: tuple[str, ...]
    storey: np.ndarray  # (N,) int16
    values: np.ndarray  # (N,) float64; all 0 for the model grid
    rgb: np.ndarray  # (N, 3) uint8; NEUTRAL_RGB for the model grid


@dataclasses.dataclass(frozen=True)
class RoomLabel:
    name: str
    storey: int
    x: float
    y: float
    z: float
    included: bool  # in the case's analysis


@dataclasses.dataclass(frozen=True, eq=False)
class Bundle:
    storeys: tuple[str, ...]  # labels, by level
    levels: tuple[float, ...]
    surfaces: dict[str, SurfaceSet]  # kind -> polygons, in KINDS order, kinds present only
    grid: PointSet
    layers: tuple[tuple[Layer, PointSet], ...]
    rooms: tuple[RoomLabel, ...]
    cell_m: float
    work_plane_m: float


def grid_points(bundle: Bundle, room: str) -> int:
    """How many model-grid points space row ``room`` has in the bundle's
    case: the one rule for "has a grid" on the atlas page (the room table's
    points and the layout sentence's grid clause). A space ticked in the
    case but not regularly occupied is included yet has none (review I1)."""
    return sum(1 for r in bundle.grid.room if r == room)


def _roof_panes(prepared: PreparedModel) -> frozenset[str]:
    """The names of every exterior pane on the roof or tilted as a
    skylight or sloped glass."""
    return frozenset(
        opening.name
        for group in prepared.groups
        for opening, pane in zip(group.openings, group.pane_classes)
        if pane.position == "roof" or pane.tilt_class in _ROOF_TILT_CLASSES
    )


def _kind(prim: Primitive, roles, roof: frozenset[str]) -> str | None:
    """Spec 6: the first matching kind; ``None`` for role ``ignore``."""
    if prim.modifier in _MODIFIER_KINDS:
        return _MODIFIER_KINDS[prim.modifier]
    role = roles.get(prim.modifier)
    if role == "ignore":
        return None
    if role is None:
        raise DeliverError(
            f"surface {prim.name!r} has modifier {prim.modifier!r}, which has no role in materials.csv"
        )
    if role == "glazing_diffusing":
        return "glass_diffusing"
    if role in EXTERIOR_GLAZING_ROLES and prim.name in roof:
        return "glass_roof"
    return _ROLE_KINDS[role]


def classify(prepared: PreparedModel) -> list[tuple[Primitive, str]]:
    """Every polygon of the prepared scene, then of every blind file, with
    its one kind (spec 6); role ``ignore`` is left out."""
    roles = prepared.effective_roles
    roof = _roof_panes(prepared)
    out: list[tuple[Primitive, str]] = []
    for prim in read_scene([prepared.scene_file], "m"):
        if prim.ptype == "polygon":
            kind = _kind(prim, roles, roof)
            if kind is not None:
                out.append((prim, kind))
    blinds = read_scene(list(prepared.blind_files.values()), "m") if prepared.blind_files else []
    out.extend((prim, "blind") for prim in blinds if prim.ptype == "polygon")
    return out


def storey_of_z(z: float, levels: Sequence[float]) -> int:
    """The storey (an index into ``levels``, ascending) whose range holds
    ``z``: from its level up to, not including, the next distinct level; the
    top one has no upper bound. At a level is in the storey above; labels
    sharing a level give the first. ``BELOW`` under the lowest level."""
    levels = [float(v) for v in levels]
    if not levels or z < levels[0] - _LEVEL_TOL_M:
        return BELOW
    best = 0
    for i, level in enumerate(levels):
        if z >= level - _LEVEL_TOL_M and level > levels[best] + _LEVEL_TOL_M:
            best = i
    return best


def _surface_set(polygons: list[tuple[np.ndarray, int]]) -> SurfaceSet:
    return SurfaceSet(
        verts=np.vstack([points for points, _ in polygons]).astype(np.float64),
        counts=np.array([points.shape[0] for points, _ in polygons], dtype=np.int32),
        storey=np.array([storey for _, storey in polygons], dtype=np.int16),
    )


def _storey_by_room(case: CaseData, rooms, index: dict[str, int]) -> dict[str, int]:
    return {room: index[case.storey_of(room)] for room in set(rooms)}


def _grid(case: CaseData, prepared: PreparedModel, index: dict[str, int]) -> PointSet:
    rooms = tuple(g.room for g in prepared.grids for _ in range(g.points.shape[0]))
    storey_of = _storey_by_room(case, rooms, index)
    xyz = np.vstack([g.points for g in prepared.grids]).astype(np.float64) if prepared.grids else np.zeros((0, 3))
    n = xyz.shape[0]
    return PointSet(
        xyz=xyz, room=rooms, storey=np.array([storey_of[r] for r in rooms], dtype=np.int16).reshape(n),
        values=np.zeros(n), rgb=np.tile(np.array(NEUTRAL_RGB, dtype=np.uint8), (n, 1)),
    )


def _layer_points(case: CaseData, layer: Layer, index: dict[str, int]) -> PointSet:
    option = case.options[layer.option]
    table = option.points
    for column in ("z", layer.column):
        if column not in table.columns:
            raise DeliverError(
                f"{option.folder / 'points.csv'}: no {column!r} column; rerun {COMMANDS[layer.option]}"
            )
    names = {space.room for space in case.spaces}
    keep = np.array([room in names for room in table.rooms], dtype=bool).reshape(len(table.rooms))
    rooms = tuple(room for room in table.rooms if room in names)
    storey_of = _storey_by_room(case, rooms, index)
    xyz = np.column_stack([table.xy[keep], np.asarray(table.columns["z"], dtype=float)[keep]])
    values = np.asarray(table.columns[layer.column], dtype=float)[keep]
    return PointSet(
        xyz=xyz, room=rooms, storey=np.array([storey_of[r] for r in rooms], dtype=np.int16).reshape(len(rooms)),
        values=values, rgb=layer_rgb(layer, values),
    )


def _included(space, case: str) -> bool:
    if case in space.include:
        return bool(space.include[case])
    return bool(space.views.get(case, False))


def build_bundle(case: CaseData, prepared: PreparedModel) -> Bundle:
    labels = tuple(storey.label for storey in case.storeys)
    levels = tuple(float(storey.level_z) for storey in case.storeys)
    index = {label: i for i, label in enumerate(labels)}
    by_kind: dict[str, list[tuple[np.ndarray, int]]] = {}
    for prim, kind in classify(prepared):
        points = prim.points
        storey = SITE if kind in _SITE_KINDS else storey_of_z(float(points[:, 2].min()), levels)
        by_kind.setdefault(kind, []).append((points, storey))
    rooms = tuple(
        RoomLabel(space.room, index[space.floor], *case.storey(space.floor).centroids[space.room],
                  float(space.level_z), _included(space, case.case))
        for space in space_rows(case.spaces)
    )
    return Bundle(
        storeys=labels, levels=levels,
        surfaces={kind: _surface_set(by_kind[kind]) for kind in KINDS if kind in by_kind},
        grid=_grid(case, prepared, index),
        layers=tuple((layer, _layer_points(case, layer, index)) for layer in result_layers(case)),
        rooms=rooms, cell_m=float(case.project.grid.spacing_m), work_plane_m=float(case.project.grid.work_plane_m),
    )


def write_npz(bundle: Bundle, path) -> None:
    """``bundle`` as the plain arrays the Blender script reads (atlas plan,
    interface contract); no object array, so ``allow_pickle=False`` reads it."""
    kinds = list(bundle.surfaces)
    arrays: dict[str, np.ndarray] = {
        "storeys": np.array(bundle.storeys, dtype=str).reshape(-1),
        "levels": np.array(bundle.levels, dtype=np.float64).reshape(-1),
        "kinds": np.array(kinds, dtype=str).reshape(-1),
        "kind_rgba": np.array([KIND_RGBA[k] for k in kinds], dtype=np.uint8).reshape(-1, 4),
    }
    for kind, surfaces in bundle.surfaces.items():
        arrays[f"k_{kind}_v"] = surfaces.verts.astype(np.float32)
        arrays[f"k_{kind}_n"] = surfaces.counts.astype(np.int32)
        arrays[f"k_{kind}_s"] = surfaces.storey.astype(np.int16)
    arrays["grid_v"] = bundle.grid.xyz.astype(np.float32).reshape(-1, 3)
    arrays["grid_s"] = bundle.grid.storey.astype(np.int16)
    arrays["grid_room"] = np.array(bundle.grid.room, dtype=str).reshape(-1)
    arrays["layer_keys"] = np.array([layer.key for layer, _ in bundle.layers], dtype=str).reshape(-1)
    arrays["layer_titles"] = np.array([layer.title for layer, _ in bundle.layers], dtype=str).reshape(-1)
    for i, (_, points) in enumerate(bundle.layers):
        arrays[f"l{i}_v"] = points.xyz.astype(np.float32).reshape(-1, 3)
        arrays[f"l{i}_s"] = points.storey.astype(np.int16)
        arrays[f"l{i}_room"] = np.array(points.room, dtype=str).reshape(-1)
        arrays[f"l{i}_value"] = points.values.astype(np.float64)
        arrays[f"l{i}_rgb"] = points.rgb.astype(np.uint8).reshape(-1, 3)
    arrays["room_names"] = np.array([r.name for r in bundle.rooms], dtype=str).reshape(-1)
    arrays["room_xyz"] = np.array([(r.x, r.y, r.z) for r in bundle.rooms], dtype=np.float64).reshape(-1, 3)
    arrays["room_s"] = np.array([r.storey for r in bundle.rooms], dtype=np.int16).reshape(-1)
    arrays["room_included"] = np.array([r.included for r in bundle.rooms], dtype=bool).reshape(-1)
    arrays["cell_m"] = np.array(bundle.cell_m, dtype=np.float64)
    arrays["work_plane_m"] = np.array(bundle.work_plane_m, dtype=np.float64)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
