"""The Quality Views model (QV spec 5.2, 6.1, 6.2): what each modifier is
to a view ray, the top of every exterior pane, the eye-height grids, and
the views scene.

The views scene is the prepared scene (frames and reveals included) with
no blinds, furniture kept or removed per analysis.moveable_furniture, and
every material replaced by a black plastic of the same name. Rays are
marched one segment at a time by the engine, so rtrace never needs to
shade anything: black materials and no light sources keep every ray cheap
while every modifier and surface name stays what the scene called it.
"""
from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np

from ..project.layout import SpaceLayout, lay_out
from ..project.materials import EXTERIOR_GLAZING_ROLES, MaterialRow
from ..project.spaces import Space, views_included
from ..scene.primitives import Primitive
from ..scene.reader import read_scene
from ._names import slug
from .glazing_class import NON_VERTICAL_TILT_CLASSES
from .grid import GridError, RoomGrid, piece_points
from .materials import copy_name
from .prepare import PreparedModel, _format_scene

_GLAZING_ROLES = EXTERIOR_GLAZING_ROLES | frozenset({"glazing_interior"})
_BLACK = (0.0, 0.0, 0.0, 0.0, 0.0)
#: Decimals a recovered VT is compared at. Radiance files carry 3-5
#: digits, so a VT of 0.40 written as a transmissivity comes back a hair
#: above it; rounding first keeps it "not above 0.40", as LEED means.
VT_DECIMALS = 4
#: Roof-glass design 5.11: every exterior pane that is not vision glazing
#: (roof glass, diffusing glass) wears this black, opaque modifier in the
#: views scene, pane by pane, so one material shared by a wall window and a
#: skylight still lets the window see out.
BLOCKED_VIEW_MODIFIER = "leed_view_blocked"


class ViewModelError(ValueError):
    """The views model cannot be built from these inputs."""


def room_file_stem(room: str) -> str:
    """The file-name stem for a room: the same slug the prepared model uses."""
    return slug(room)


def view_classes(
    rows: Sequence[MaterialRow], visible_transmittance: Mapping[str, float], *,
    vlt_min_exclusive: float,
) -> dict[str, str]:
    """QV spec 5.2: modifier -> clear | blocks | nature | art | landmark.

    Default: a glazing role whose recovered visible transmittance, rounded
    to VT_DECIMALS, is above the limit is clear; everything else blocks. An
    explicit `view` wins, except `clear` on glass at or below the limit,
    which LEED forbids. Diffusing glazing always blocks (roof-glass design
    5.11; materials.csv allows it no view tag but `blocks`), and so does
    frosted interior glass, a `trans` with role `glazing_interior` (QV spec
    5.2: only `glass` or `dielectric` can be clear); a `view` on it other
    than empty or `blocks` is refused here too, so the refusal also holds
    for a row whose `primitive_type` the views driver filled from the scene.
    """
    classes: dict[str, str] = {}
    for row in rows:
        frosted = row.role == "glazing_interior" and row.primitive_type == "trans"
        if frosted and row.view not in ("", "blocks"):
            raise ViewModelError(
                f"{row.modifier}: a 'trans' with role 'glazing_interior' is frosted glass, never "
                f"vision glazing, so its view must be empty or 'blocks', got {row.view!r}"
            )
        if row.role == "glazing_diffusing" or frosted:
            classes[row.modifier] = "blocks"
            continue
        vt = visible_transmittance.get(row.modifier)
        if vt is not None:
            vt = round(float(vt), VT_DECIMALS)
        if row.view:
            if row.view == "clear" and row.role in _GLAZING_ROLES and vt is not None and not vt > vlt_min_exclusive:
                raise ViewModelError(
                    f"{row.modifier}: marked 'clear' in materials.csv, but its visible "
                    f"transmittance {vt:.4f} is not above {vlt_min_exclusive:.2f}; LEED "
                    "requires glass with a VLT above that limit"
                )
            classes[row.modifier] = row.view
        elif row.role in _GLAZING_ROLES and vt is not None and vt > vlt_min_exclusive:
            classes[row.modifier] = "clear"
        else:
            classes[row.modifier] = "blocks"
    return classes


def copy_rows(rows: Sequence[MaterialRow], effective_roles: Mapping[str, str]) -> list[MaterialRow]:
    """One synthetic materials.csv row per material copy (roof-glass design
    5.4 step 3): the copy's name with its original's type, role, note and
    view tag, so a sloped perimeter copy is classed exactly like its
    original. A materials.csv modifier that merely looks like a copy name
    is a material of its own and keeps its own row."""
    return [
        dataclasses.replace(row, modifier=name)
        for row, name in _copies(rows)
        if name in effective_roles
    ]


def _copies(rows: Sequence[MaterialRow]) -> list[tuple[MaterialRow, str]]:
    """Every (original row, copy name) pair whose copy name is not itself a
    materials.csv modifier."""
    listed = {row.modifier for row in rows}
    return [
        (row, name)
        for row in rows
        for name in (copy_name(row.modifier, c) for c in NON_VERTICAL_TILT_CLASSES)
        if name not in listed
    ]


def vision_panes(prepared: PreparedModel) -> frozenset[str]:
    """Exterior pane names that are vision glazing (roof-glass design 5.11,
    D7): at position ``perimeter`` and not diffusing. A group with no pane
    classes is all vision, as every group was before roof glass existed."""
    names: set[str] = set()
    for group in prepared.groups:
        classes = group.pane_classes or (None,) * len(group.openings)
        for opening, pane_class in zip(group.openings, classes):
            if pane_class is None or (pane_class.position == "perimeter" and not pane_class.diffusing):
                names.add(opening.name)
    return frozenset(names)


def blocked_panes(prepared: PreparedModel) -> frozenset[str]:
    """Every other exterior pane name: roof glass and diffusing glass."""
    vision = vision_panes(prepared)
    return frozenset(o.name for g in prepared.groups for o in g.openings if o.name not in vision)


def blocked_clear_warnings(prepared: PreparedModel, rows: Sequence[MaterialRow]) -> list[str]:
    """One warning per materials.csv material tagged `view: clear` that a
    blocked pane uses, directly or through its tilt-class copy; the pane
    named is the first by name."""
    view_of = {row.modifier: row.view for row in rows}
    original = {name: row.modifier for row, name in _copies(rows)}
    blocked = blocked_panes(prepared)
    panes: dict[str, list[str]] = {}
    for group in prepared.groups:
        for opening in group.openings:
            if opening.name not in blocked:
                continue
            material = original.get(opening.modifier, opening.modifier)
            if view_of.get(material) == "clear":
                panes.setdefault(material, []).append(opening.name)
    return [
        f"{material}: marked 'clear' in materials.csv, but pane {min(names)} uses it and is "
        "roof or diffusing glass, which is not vision glazing; that pane blocks views"
        for material, names in sorted(panes.items())
    ]


def pane_tops(prepared: PreparedModel) -> dict[str, float]:
    """Vision pane surface name -> the top z of its framed glass (QV spec
    6.5). Roof and diffusing glass set no head height (roof-glass design
    5.11)."""
    vision = vision_panes(prepared)
    tops: dict[str, float] = {}
    for group in prepared.groups:
        for opening, glass in zip(group.openings, prepared.glass_by_group[group.group_id]):
            if opening.name in vision:
                tops[opening.name] = float(max(point[2] for point in glass))
    return tops


def ground_plane(
    prims: Sequence[Primitive], roles: Mapping[str, str], *,
    storey_levels: Sequence[float], room_levels: Mapping[str, float],
) -> tuple[float, list[str]]:
    """The z a downward ray leaving the scene lands on (QV spec 6.4), and
    the warnings it carries.

    The lowest z of any ground-role polygon. Without one, the lowest
    storey level: a model in datum heights (floors near +100 m) would
    otherwise measure every downward view to z = 0 and pass Type 2 on
    distances that are not there. A room whose floor is below the plane
    used is named.
    """
    lows = [
        float(p.points[:, 2].min())
        for p in prims
        if p.ptype == "polygon" and roles.get(p.modifier) == "ground"
    ]
    warnings: list[str] = []
    if lows:
        z = min(lows)
    elif storey_levels:
        z = float(min(storey_levels))
        warnings.append(f"no ground surface: ground taken at the lowest floor level, z = {z:.2f}")
    else:
        raise ViewModelError("no ground surface and no floor level; cannot place the ground plane")
    for room, level in room_levels.items():
        if level < z - 1e-6:
            warnings.append(
                f"room {room!r}: floor level z = {level:.2f} is below the ground plane "
                f"z = {z:.2f}; downward views from it are measured to that plane"
            )
    return z, warnings


def view_grids(
    spaces: Sequence[Space], matched: Mapping[str, object], grid, eye_height_m: float, case: str,
    *, layout: SpaceLayout | None = None,
) -> list[RoomGrid]:
    """One eye-height grid per space ticked in views_<case> (QV spec 6.2):
    its pieces' points, as ``build.grid.build_grids`` lays them, at eye
    height. ``layout`` is the space layout; without one it is laid out
    here from ``spaces`` and ``matched``."""
    eye_grid = dataclasses.replace(grid, work_plane_m=eye_height_m)
    grids: list[RoomGrid] = []
    selected = views_included(spaces, case)
    for space in selected:
        if matched.get(space.room) is None:
            raise ViewModelError(f"space {space.room!r} is ticked in views_{case} but matches no room")
    if layout is None:
        layout = lay_out(spaces, matched)
    for space in selected:
        laid = layout.space(space.room)
        try:
            pieces = piece_points(laid, eye_grid)
        except GridError as exc:
            raise ViewModelError(str(exc)) from None
        if len(laid.pieces) == 1 and pieces[0].shape[0] == 0:
            raise ViewModelError(
                f"room {space.room!r} (area {laid.area_m2:.2f} m2) yields no view point at "
                f"{grid.spacing_m:g} m spacing"
            )
        for piece, row, points in zip(laid.pieces, laid.rows, pieces, strict=True):
            if points.shape[0] == 0:
                raise ViewModelError(
                    f"room {space.room!r}: piece {piece.room_id} (row {row!r}, area "
                    f"{piece.area_m2:.2f} m2) yields no view point at {grid.spacing_m:g} m "
                    "spacing; every piece of a space needs at least one view point; clear its "
                    "merge_into and untick it, or check the room in the model"
                )
        grids.append(RoomGrid(
            room=space.room,
            points=np.concatenate(pieces) if len(pieces) > 1 else pieces[0],
            spacing_m=grid.spacing_m,
            cell_area_m2=grid.spacing_m ** 2,
            room_area_m2=laid.area_m2,
            storey=laid.storey,
            level_z=laid.level_z,
        ))
    return grids


def write_view_scene(
    prepared: PreparedModel, roles: Mapping[str, str], *, moveable_furniture: str, out_dir,
    blocked: frozenset[str] = frozenset(),
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the views scene and its black materials (QV spec 6.1). Every
    surface named in ``blocked`` wears BLOCKED_VIEW_MODIFIER (roof-glass
    design 5.11)."""
    if moveable_furniture not in ("include", "exclude"):
        raise ViewModelError(f"moveable_furniture must be include or exclude, got {moveable_furniture!r}")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces = read_scene([prepared.scene_file], "m")
    materials = [
        Primitive("void", "plastic", p.name, (), (), _BLACK, "build.view_model")
        for p in read_scene([prepared.materials_file], "m")
        if not p.is_surface
    ]
    if blocked:
        if BLOCKED_VIEW_MODIFIER in {p.name for p in surfaces} | {p.name for p in materials}:
            raise ViewModelError(
                f"{BLOCKED_VIEW_MODIFIER!r}: the scene already uses this name; the views model "
                "needs it for the roof and diffusing glass that blocks views. Rename it in the scene"
            )
        surfaces = [
            dataclasses.replace(p, modifier=BLOCKED_VIEW_MODIFIER) if p.name in blocked else p
            for p in surfaces
        ]
        materials.append(Primitive("void", "plastic", BLOCKED_VIEW_MODIFIER, (), (), _BLACK, "build.view_model"))
    if moveable_furniture == "exclude":
        surfaces = [p for p in surfaces if roles.get(p.modifier) != "furniture"]
    scene_file = out_dir / "views_scene.rad"
    materials_file = out_dir / "views_materials.rad"
    scene_file.write_text(_format_scene(surfaces), encoding="utf-8", newline="\n")
    materials_file.write_text(_format_scene(materials), encoding="utf-8", newline="\n")
    return scene_file, materials_file
