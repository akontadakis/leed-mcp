# leed_daylight/build/prepare.py
"""Compose the prepared model: scene, materials, blinds, grids and manifest.

This is the last step before Radiance ever runs (layer 4, not started here).
It composes every layer-2 and layer-3 module that came before it into one
folder a person -- or phase 4 -- can point a Radiance command at, plus a
manifest that makes the run auditable: every recovered glazing value, every
solved frame, every group and grid, and a SHA-256 of every file this run
wrote, so a re-run can be told apart from a changed one. The manifest never
hashes itself (a self-digest cannot exist).

Build order (spec 5.4 of the skylights design): validate the scene as given
-> match spaces to rooms and form window groups on the ORIGINAL primitives,
which classifies every exterior pane (tilt class, position, diffusing) ->
check what the classes require (-ab for diffusing glass, the skylight well)
-> rebuild every material, with a copy per sloped or skylight use of a
glazing material -> for every exterior opening, solve its frame (share by
class) and its reveal or skylight well, and write the pane back once with
its framed glass and its new modifier -> build one blind per group except
diffusing groups -> build the sensor grid. Interior glazing
(``glazing_interior``) never reaches a window group, so it never gets a
frame, a reveal or a blind; its rebuilt glass stays in the scene exactly
where the modeller put it (spec 7's integration test, complete).

Spaces (merge-split spec 4, 7, 8): the spaces are matched to the rooms
(split halves included) and laid out (``project.layout.lay_out``); window
groups are formed per piece, a reveal is clipped at the floor and ceiling
of its opening's own piece, and each space gets one grid over all its
pieces. The manifest gains a ``layout`` key only when a space of the case
is not a plain detected room, so a project with no split and no merge
prepares byte for byte as before. With ``grid.points_file`` the grids are
the file's points (``build.grid.build_grids``) and the manifest gains
``grid_source: points_file``; without it nothing changes.

No generated primitive name may collide with a name already in the scene:
a collision is refused by name rather than silently shadowing whichever
primitive was there first. Nothing is written to disk until every check has
passed, so a refusal never leaves a half-written folder behind.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import shlex
import tempfile
import types
from collections.abc import Mapping, Sequence

import numpy as np
from shapely.geometry import Point, Polygon

from .. import geometry
from ..constants import TOOL_VERSION
from ..project.layout import LaidOutSpace, SpaceLayout, SplitHalf, lay_out
from ..project.points import read_points
from ..project.spaces import (
    Space,
    area_mismatches,
    check_linked_zones,
    included,
    match_to_rooms,
    piece_row_warnings,
)
from ..scene.primitives import Primitive
from ..scene.reader import quote_word
from ..scene.storeys import Room, RoomDetection
from ..scene.validate import validate_scene
from ._names import slug as _slug
from .blinds import blind_material, blind_polygons
from .frames import build_frame, drop_redundant_vertices
from .glazing_class import NON_VERTICAL_TILT_CLASSES, PaneClass
from .grid import RoomGrid, build_grids
from .groups import LOOKDOWN_ROLES, WindowGroup, structure_polygons, window_groups
from .lookdown import first_hit_below
from .materials import MaterialError, rebuild_materials_by_class
from .reveals import (
    RevealError,
    reveal_walls,
    shared_edges,
    skylight_well_walls,
    well_shared_edges,
)

_FRAME_MATERIAL_NAME = "leed_frame"
_REVEAL_MATERIAL_NAME = "leed_reveal"
_WELL_MATERIAL_NAME = "leed_skylight_well"

#: Spec 4.3 / 5.9: light through a diffuser reaches a sensor only through
#: ambient rays, one bounce more than through clear glass.
_MIN_DIFFUSING_AB: int = 2

#: merge-split Review Focus 1: appended to a spaces.csv matching refusal
#: when split lines changed the rooms, since a row written before the split
#: still carries the whole room's centroid.
_SPLIT_HINT: str = (
    "; splits.csv changed the rooms: run the rooms command again and copy each "
    "half's centroid into spaces.csv"
)


def split_hint(rooms: Sequence[Room]) -> str:
    """The hint ``match_to_rooms`` appends to its refusals: ``_SPLIT_HINT``
    when split lines changed ``rooms``, else ``""``. ``prepare`` and
    ``deliver.data.prepare_case_model`` (which the atlas runs first) both
    match with it (review M1)."""
    return _SPLIT_HINT if any(isinstance(room, SplitHalf) for room in rooms) else ""


class PrepareError(ValueError):
    """The prepared model cannot be built or written from these inputs."""


@dataclasses.dataclass(frozen=True)
class PreparedModel:
    root: pathlib.Path
    case: str
    scene_file: pathlib.Path
    materials_file: pathlib.Path
    blind_files: dict[str, pathlib.Path]  # group_id -> .rad, one file per group
    grid_files: dict[str, pathlib.Path]  # room -> .pts
    manifest_path: pathlib.Path
    manifest: dict
    #: Additive (controller ruling P26): so phase 4 can consume a prepared
    #: model without reopening any phase-3 module. Sorted by group_id.
    groups: tuple[WindowGroup, ...]
    grids: tuple[RoomGrid, ...]
    #: Each group's FRAMED glass outlines (after the inset), aligned with
    #: that group's own ``openings`` -- a phase-4 sender never has to
    #: subtract the frame itself to find the glass.
    glass_by_group: Mapping[str, tuple[np.ndarray, ...]]
    #: Every modifier's role, plus one entry per sloped/skylight material
    #: copy carrying its original's role (spec 5.4 step 5). Every consumer
    #: that looks a modifier's role up after prepare uses this map.
    effective_roles: types.MappingProxyType = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))
    #: The spaces laid out over the rooms (merge-split spec 6).
    layout: SpaceLayout | None = None


def _check_unique_surface_names(prims: Sequence[Primitive]) -> None:
    """Refuse two surfaces sharing a name, naming it and both positions (C1).

    Radiance allows repeated primitive names; this tool does not, because
    every place downstream that looks a surface up by name (this module's
    own ``scene.rad`` assembly used to, before the fix) would silently keep
    only the LAST surface written under that name and lose the other's
    geometry. Checked before anything else runs, on every surface in the
    scene -- not only glazing openings -- so a repeated wall or partition
    name is caught here rather than surfacing as a misleading generated-name
    collision three steps later.
    """
    seen: dict[str, int] = {}
    for i, p in enumerate(prims):
        if not p.is_surface:
            continue
        if p.name in seen:
            raise PrepareError(
                f"{p.name!r}: a surface name is used more than once, at "
                f"position {seen[p.name]} and position {i}; Radiance allows "
                "repeated names but this tool does not, since rebuilding "
                "the scene would silently keep only one surface's geometry "
                "under that name"
            )
        seen[p.name] = i


def _check_grid_slug_collisions(grids: Sequence[RoomGrid]) -> None:
    """Refuse two spaces whose grid file names collide, naming both (I4).

    ``build.groups`` already refuses a slug collision between two window
    groups' rooms; a space can also get a grid without ever appearing in a
    window group (a room with no exterior glazing still gets a sensor
    grid), so that check alone does not cover every space a slug collision
    can silently overwrite here.
    """
    owner_by_slug: dict[str, str] = {}
    for room_grid in grids:
        s = _slug(room_grid.room)
        owner = owner_by_slug.setdefault(s, room_grid.room)
        if owner != room_grid.room:
            raise PrepareError(
                f"{s!r}: space {room_grid.room!r} and space {owner!r} both "
                f"slug to the same grid file name {s!r}.pts; rename one so "
                "grid files stay distinct"
            )


def _flatten(points) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(points, dtype=np.float64).reshape(-1))


def _format_primitive(p: Primitive) -> str:
    """One primitive, in Radiance text: ``modifier ptype name`` then the
    three argument lines. Floats are written with ``repr()`` (the shortest
    string that round-trips), so the same primitive always serialises to
    the same bytes. Every word is written through
    :func:`scene.reader.quote_word`, so a word holding a space, a ``#`` or
    a quote reads back as the same word."""
    lines = [
        " ".join(quote_word(w) for w in (p.modifier, p.ptype, p.name)),
        " ".join([str(len(p.sargs)), *(quote_word(w) for w in p.sargs)]),
        " ".join([str(len(p.iargs)), *(str(i) for i in p.iargs)]),
        " ".join([str(len(p.fargs)), *(repr(float(f)) for f in p.fargs)]),
    ]
    return "\n".join(lines) + "\n"


def _format_scene(prims: Sequence[Primitive]) -> str:
    return "".join(_format_primitive(p) for p in prims)


def _format_pts(points: np.ndarray) -> str:
    lines = [
        " ".join((repr(float(x)), repr(float(y)), repr(float(z)), "0", "0", "1"))
        for x, y, z in points
    ]
    return "".join(line + "\n" for line in lines)


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    """Write ``data`` to a temporary file beside ``path``, then rename it
    over ``path``: a reader sees the old file or the new one, never half."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        # mkstemp creates the file owner-only; give it the permissions an
        # ordinary write would have had.
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(tmp_name, 0o666 & ~mask)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


class _NameRegistry:
    """Every primitive name already in the scene; refuses a generated name
    that collides with one, naming it."""

    def __init__(self, existing: Sequence[str]):
        self._names = set(existing)

    def claim(self, name: str) -> None:
        if name in self._names:
            raise PrepareError(
                f"{name!r}: a generated primitive name collides with one "
                "already in the scene"
            )
        self._names.add(name)


def _frame_share_for(
    role: str,
    frame_share: Mapping[str, float],
    tilt_class: str = "vertical",
    pane: str = "",
) -> float:
    """Spec 5.5: a sloped or skylight pane takes its class key, which is
    required; every other pane its role key, else ``default``."""
    if tilt_class in NON_VERTICAL_TILT_CLASSES:
        if tilt_class in frame_share:
            return frame_share[tilt_class]
        raise PrepareError(
            f"glazing.frame_share: no {tilt_class!r} entry, but pane {pane} is "
            f"{tilt_class} class; a {tilt_class} pane takes its frame share "
            f"from glazing.frame_share.{tilt_class}, never from its role or "
            "'default'"
        )
    if role in frame_share:
        return frame_share[role]
    if "default" in frame_share:
        return frame_share["default"]
    raise PrepareError(
        f"glazing.frame_share: no entry for role {role!r} and no 'default' "
        "entry either; every exterior glazing role needs a frame share"
    )


def _check_ambient(radiance, pane_classes: Mapping[str, PaneClass]) -> None:
    """Spec 4.3: with any diffusing pane, the last ``-ab`` in
    ``radiance.ambient.full`` followed by ``radiance.direct`` (the engine
    puts both on one command line in that order) must be 2 or more.

    Only the separate-token form ``-ab N`` is read, as the engine writes
    it; ``-ab`` with no whole number after it is refused, as is an options
    file (``@file``), whose ``-ab`` cannot be seen."""
    diffusing = [name for name, pc in pane_classes.items() if pc.diffusing]
    if not diffusing:
        return
    pane = diffusing[0]
    need = f"diffusing glazing (pane {pane}) needs -ab {_MIN_DIFFUSING_AB} or more"
    # Each refusal names the option its offending token sits in.
    tokens: list[str] = []
    source: list[str] = []
    for option, text in (
        ("radiance.ambient.full", radiance.ambient_full),
        ("radiance.direct", radiance.direct),
    ):
        try:
            words = shlex.split(text)
        except ValueError as exc:
            raise PrepareError(
                f"{option}: cannot be read as options ({exc}); {need}"
            ) from None
        tokens += words
        source += [option] * len(words)
    for token, option in zip(tokens, source):
        if token.startswith("@"):
            raise PrepareError(
                f"{option}: {token!r} reads an options file whose "
                f"-ab cannot be checked; {need}, so write the options out"
            )
    last: int | None = None
    last_option = ""
    for i, token in enumerate(tokens):
        if token != "-ab":
            continue
        try:
            last = int(tokens[i + 1])
        except (IndexError, ValueError):
            raise PrepareError(
                f"{source[i]}: -ab needs a whole number after it; {need}"
            ) from None
        last_option = source[i]
    if last is None:
        raise PrepareError(
            f"radiance.ambient.full: {need}, but neither radiance.ambient.full "
            "nor radiance.direct sets -ab (rcontrib's default is 1)"
        )
    if last < _MIN_DIFFUSING_AB:
        raise PrepareError(
            f"{last_option}: {need}; the last -ab in "
            f"radiance.ambient.full then radiance.direct is {last}"
        )


def _well_depth(glazing, pane_classes: Mapping[str, PaneClass]) -> float | None:
    """The depth of the wells to build, or ``None`` (no skylight-class
    glass, or ``modelled``). Refuses a missing ``skylight_well`` when
    skylight-class glass exists (spec 4.2)."""
    skylights = [name for name, pc in pane_classes.items() if pc.tilt_class == "skylight"]
    if not skylights:
        return None
    if glazing.skylight_well is None:
        raise PrepareError(
            f"glazing.skylight_well: missing, but pane {skylights[0]} is "
            "skylight class; give 'modelled' (the scene carries the well) or "
            "a well depth in metres"
        )
    if glazing.skylight_well == "modelled":
        return None
    return float(glazing.skylight_well)


def _well_manifest(glazing):
    """The manifest's top-level ``skylight_well`` (spec 7): ``"modelled"``,
    the depth and reflectance, or ``None`` when the key is not set."""
    well = glazing.skylight_well
    if well is None or well == "modelled":
        return well
    return {"depth_m": float(well), "reflectance": glazing.skylight_well_reflectance}


def _well_skips(
    groups: Sequence[WindowGroup], clean_outline: Mapping[int, np.ndarray], layout: SpaceLayout
) -> dict[int, frozenset[int]]:
    """Spec 5.6: per skylight-class pane, the edges it shares with another
    skylight-class pane of the same room (a ridge or hip gets no well
    wall), keyed by the pane's identity. The panes of one space count as
    one room, and so do the spaces holding halves cut from one detected
    room (merge-split Open point 13): a split may run along a ridge, which
    the unsplit room had no well wall on."""
    parent: dict[tuple, tuple] = {}

    def find(key: tuple) -> tuple:
        while parent.get(key, key) != key:
            key = parent[key]
        return key

    for group in groups:
        piece = layout.space(group.room).pieces[group.piece]
        if isinstance(piece, SplitHalf):
            own, shared = find(("space", group.room)), find(("room", id(piece.source_outline)))
            if own != shared:
                parent[own] = shared
    by_room: dict[tuple, list[Primitive]] = {}
    for group in groups:
        for opening, pane_class in zip(group.openings, group.pane_classes):
            if pane_class.tilt_class == "skylight":
                by_room.setdefault(find(("space", group.room)), []).append(opening)
    skips: dict[int, frozenset[int]] = {}
    for panes in by_room.values():
        try:
            shared = well_shared_edges([clean_outline[id(o)] for o in panes])
        except RevealError as exc:
            raise PrepareError(f"skylight wells: {exc}") from None
        skips.update({id(o): s for o, s in zip(panes, shared)})
    return skips


def _point_inside_pane(points: np.ndarray) -> np.ndarray:
    """A point on a skylight-class pane whose plan position lies inside the
    pane's plan outline: the centroid when it does, else shapely's
    ``representative_point`` of the plan outline, lifted onto the pane's
    own plane. A skylight-class pane is under 20 degrees from flat, so its
    plan outline is a proper polygon and its normal has a z component."""
    centroid = geometry.centroid(points)
    plan = Polygon(points[:, :2])
    if plan.covers(Point(float(centroid[0]), float(centroid[1]))):
        return centroid
    inside = plan.representative_point()
    x, y = float(inside.x), float(inside.y)
    n = geometry.normal(points)
    z = float(centroid[2]) - (
        float(n[0]) * (x - float(centroid[0])) + float(n[1]) * (y - float(centroid[1]))
    ) / float(n[2])
    return np.array([x, y, z])


def _ceiling_warnings(
    prims: Sequence[Primitive],
    roles: Mapping[str, str],
    groups: Sequence[WindowGroup],
    clean_outline: Mapping[int, np.ndarray],
) -> tuple[str, ...]:
    """Spec 5.6: warn when a ray straight down from a roof skylight
    meets a ceiling-role polygon before anything else of the room's
    structure (a floor ends the search). A warning, never a refusal: a
    deliberate diffuser ceiling may be modelled as ``ceiling``.

    The ray starts at a point inside the pane (``_point_inside_pane``):
    the centroid whenever it lies inside the pane's plan outline, as it
    always does for a convex pane, else a point inside a concave one
    (an L), whose centroid can sit over solid ceiling beside the hole."""
    skylights = [
        opening
        for group in groups
        for opening, pane_class in zip(group.openings, group.pane_classes)
        if pane_class.tilt_class == "skylight" and pane_class.position == "roof"
    ]
    if not skylights:
        return ()
    polygons = structure_polygons(prims, roles)
    role_by_name = {
        p.name: roles[p.modifier]
        for p in prims
        if p.is_surface and roles.get(p.modifier) in LOOKDOWN_ROLES
    }
    warnings: list[str] = []
    for opening in skylights:
        origin = _point_inside_pane(clean_outline[id(opening)])
        hit = first_hit_below(origin, polygons)
        if hit is not None and role_by_name[hit[0]] == "ceiling":
            warnings.append(
                f"a ceiling surface lies under skylight {opening.name}; "
                "its light is blocked"
            )
    return tuple(warnings)


def _layout_entry(space: LaidOutSpace) -> dict:
    """The manifest's record of one space that is not a plain detected
    room: its origin and each piece's room id and area, with the
    splits.csv rows of a split half."""
    pieces = []
    for piece in space.pieces:
        entry = {"room_id": piece.room_id, "area_m2": float(piece.area_m2)}
        if isinstance(piece, SplitHalf):
            entry["splits_rows"] = [cut.row for cut in piece.cuts]
        pieces.append(entry)
    return {"origin": space.origin, "pieces": pieces}


def prepare(
    project,
    prims: Sequence[Primitive],
    roles: Mapping[str, str],
    rooms: RoomDetection,
    spaces: Sequence[Space],
    case: str,
    out_dir,
    *,
    check_ambient: bool = True,
) -> PreparedModel:
    """Build the prepared model for one credit-scenario case, or refuse.

    ``rooms`` is the :class:`~leed_daylight.scene.storeys.RoomDetection`
    from ``project.layout.scene_rooms`` (``detect_rooms`` and the split
    lines), so its own warnings (an excluded floor) reach the manifest
    rather than being lost. ``prepare`` validates the scene itself
    (``validate_scene``); any error is raised as :class:`SceneError` before
    anything is written. It also matches ``spaces`` onto ``rooms.rooms``
    itself, via :func:`match_to_rooms`, and lays the spaces out over them
    (``project.layout.lay_out``).

    ``check_ambient=False`` skips the ``-ab`` check for diffusing glazing
    (spec 4.3); only the views driver passes it, since it traces with
    ``-ab 0`` by design.
    """
    prims = list(prims)
    out_dir = pathlib.Path(out_dir).absolute()
    if out_dir.exists() and not out_dir.is_dir():
        raise PrepareError(
            f"out_dir {out_dir}: exists and is not a directory; refusing to "
            "write a prepared model into it"
        )

    _check_unique_surface_names(prims)

    report = validate_scene(prims, project.scene.units, roles)
    report.raise_if_errors()

    selected = included(spaces, case)

    # Spec 5.4 step 2: classify, match and group every exterior pane on the
    # ORIGINAL primitives. Step 3 replaces material primitives only; every
    # surface passes through it as the same object, so the openings held
    # by `groups` are the very objects `position_by_id` indexes below.
    # Panes are matched to the pieces of each space (merge-split spec 7).
    name_to_room = match_to_rooms(spaces, rooms.rooms, hint=split_hint(rooms.rooms))
    layout = lay_out(spaces, name_to_room)
    check_linked_zones(spaces, project.performance.linked_zones)
    area_warnings = area_mismatches(spaces, name_to_room)
    groups, group_warnings = window_groups(
        prims, roles, layout.pieces_by_space(), all_rooms=rooms.rooms,
    )

    if not groups:
        raise PrepareError(
            "no exterior glazing found in this scene; spec 6.0 requires at "
            "least one exterior glazing opening before a model can be "
            "prepared"
        )

    pane_classes = {
        opening.name: pane_class
        for group in groups
        for opening, pane_class in zip(group.openings, group.pane_classes, strict=True)
    }
    # Spec 5.12: the new manifest keys only when some group is not a window.
    extended = any(group.kind != "window" for group in groups)
    if check_ambient:
        _check_ambient(project.radiance, pane_classes)
    well_depth = _well_depth(project.glazing, pane_classes)

    # Spec 5.4 step 3. The registry holds the ORIGINAL names; each copy
    # name is claimed, so a copy clashing with any scene name is refused.
    names = _NameRegistry(p.name for p in prims)
    try:
        classed = rebuild_materials_by_class(prims, roles, project, pane_classes)
    except MaterialError as exc:
        raise PrepareError(str(exc)) from None
    for record in classed.records:
        if record.tilt_class != "vertical":
            names.claim(record.modifier)
    rebuilt = classed.prims
    glazing_records = classed.records
    materials_warnings = classed.warnings
    pane_modifier = classed.pane_modifier
    effective_roles = classed.effective_roles

    # C1: an opening's rebuilt glass replaces the original IN PLACE, by its
    # own list position in `rebuilt` -- never by name. Radiance allows two
    # surfaces to share a name; looking one back up by name after this
    # point would silently keep only the last surface written under it.
    scene_list: list[Primitive] = list(rebuilt)
    position_by_id: dict[int, int] = {id(p): i for i, p in enumerate(rebuilt)}

    generated_surfaces: list[Primitive] = []
    generated_materials: list[Primitive] = []
    opening_records: dict[str, dict] = {}
    blind_prims_by_group: dict[str, list[Primitive]] = {}
    glass_by_group: dict[str, tuple[np.ndarray, ...]] = {}

    uses_reveal = project.glazing.reveal_depth_m > 0.0

    names.claim(_FRAME_MATERIAL_NAME)
    r = project.glazing.frame_reflectance
    generated_materials.append(
        Primitive("void", "plastic", _FRAME_MATERIAL_NAME, (), (), (r, r, r, 0.0, 0.0), "build.prepare")
    )
    if uses_reveal:
        names.claim(_REVEAL_MATERIAL_NAME)
        rr = project.glazing.reveal_reflectance
        generated_materials.append(
            Primitive("void", "plastic", _REVEAL_MATERIAL_NAME, (), (), (rr, rr, rr, 0.0, 0.0), "build.prepare")
        )
    if well_depth is not None:
        names.claim(_WELL_MATERIAL_NAME)
        wr = project.glazing.skylight_well_reflectance
        generated_materials.append(
            Primitive("void", "plastic", _WELL_MATERIAL_NAME, (), (), (wr, wr, wr, 0.0, 0.0), "build.prepare")
        )
    blind_mat = blind_material(project.blinds)
    names.claim(blind_mat.name)
    generated_materials.append(blind_mat)

    # Build order (spec 5.4 to 5.8): for each exterior opening, solve its
    # frame, then its skylight well (a skylight-class pane, when a well
    # depth is set; none when the well is modelled) or its reveal (any
    # other pane, when reveals are on; room side only for sloped roof
    # glass), THEN build the group's blinds -- except for a diffusing
    # group, which gets no blind. Blinds are offset from the opening's own
    # outline, which (like the group itself) does not depend on the frame,
    # so the order follows the spec's sequence, not a data dependency.
    # Reveal inputs: every exterior opening's outline without repeated or
    # collinear vertices (M1), and which of its edges are joints with a
    # coplanar neighbouring pane (a curtain wall's mullions), which get no
    # reveal.
    clean_outline: dict[int, np.ndarray] = {
        id(o): drop_redundant_vertices(o.points) for g in groups for o in g.openings
    }
    joint_edges: dict[int, frozenset[int]] = {}
    if uses_reveal:
        all_openings = [o for g in groups for o in g.openings]
        try:
            joints = shared_edges([clean_outline[id(o)] for o in all_openings])
        except RevealError as exc:
            raise PrepareError(f"reveals: {exc}") from None
        joint_edges = {id(o): j for o, j in zip(all_openings, joints)}
    well_skip: dict[int, frozenset[int]] = (
        _well_skips(groups, clean_outline, layout) if well_depth is not None else {}
    )
    ceiling_warnings = _ceiling_warnings(prims, roles, groups, clean_outline)

    for group in groups:
        group_glass: list[np.ndarray] = []
        # merge-split spec 7: the floor and ceiling of the opening's own piece.
        room = layout.space(group.room).pieces[group.piece]
        floor_z = room.level_z
        ceiling_z = (
            None if room.ceiling_height_m is None else room.level_z + room.ceiling_height_m
        )
        for opening, outward, pane_class in zip(
            group.openings, group.outward_normals, group.pane_classes, strict=True
        ):
            modifier = pane_modifier.get(opening.name, opening.modifier)
            role = effective_roles.get(modifier)
            share = _frame_share_for(
                role, project.glazing.frame_share, pane_class.tilt_class, opening.name
            )
            frame = build_frame(opening.points, opening.name, share, project.glazing.frame_style)

            # Spec 5.4 step 4: the pane is written back ONCE, with its framed
            # glass and its new modifier in the same replace.
            scene_list[position_by_id[id(opening)]] = dataclasses.replace(
                opening, modifier=modifier, fargs=_flatten(frame.glass)
            )
            group_glass.append(frame.glass)

            for i, quad in enumerate(frame.ring):
                quad_name = f"{opening.name}_frame_{i}"
                names.claim(quad_name)
                generated_surfaces.append(
                    Primitive(_FRAME_MATERIAL_NAME, "polygon", quad_name, (), (), _flatten(quad), "build.prepare")
                )

            # Spec 5.6: geometry by (tilt class, position), never by kind.
            if pane_class.tilt_class == "skylight":
                # A skylight gets a well (or nothing when it is modelled),
                # never a reveal.
                if well_depth is not None:
                    try:
                        well_quads = skylight_well_walls(
                            clean_outline[id(opening)],
                            well_depth,
                            skip_edges=well_skip[id(opening)],
                        )
                    except RevealError as exc:
                        raise PrepareError(f"{opening.name}: {exc}") from None
                    for i, quad in enumerate(well_quads):
                        quad_name = f"{opening.name}_well_{i}"
                        names.claim(quad_name)
                        generated_surfaces.append(
                            Primitive(_WELL_MATERIAL_NAME, "polygon", quad_name, (), (), _flatten(quad), "build.prepare")
                        )
            elif uses_reveal:
                try:
                    reveal_quads = reveal_walls(
                        clean_outline[id(opening)],
                        outward,
                        project.glazing.reveal_depth_m,
                        floor_z=floor_z,
                        ceiling_z=ceiling_z,
                        skip_edges=joint_edges[id(opening)],
                        inner_only=(
                            pane_class.position == "roof" and pane_class.tilt_class == "sloped"
                        ),
                    )
                except RevealError as exc:
                    raise PrepareError(f"{opening.name}: {exc}") from None
                for i, quad in enumerate(reveal_quads):
                    quad_name = f"{opening.name}_reveal_{i}"
                    names.claim(quad_name)
                    generated_surfaces.append(
                        Primitive(_REVEAL_MATERIAL_NAME, "polygon", quad_name, (), (), _flatten(quad), "build.prepare")
                    )

            opening_record = {
                "modifier": modifier,
                "role": role,
                "target_glass_area_m2": frame.target_glass_area_m2,
                "achieved_glass_area_m2": frame.achieved_glass_area_m2,
                "frame_width_m": frame.width_m,
                "frame_style": project.glazing.frame_style,
                "frame_share": share,
            }
            if extended:
                opening_record["class"] = dataclasses.asdict(pane_class)
            opening_records[opening.name] = opening_record

        glass_by_group[group.group_id] = tuple(group_glass)

        if group.kind == "diffusing":
            # Spec 5.8: diffusing glass gets no blind and no blind file.
            continue

        # M1: the blind follows the outline without repeated or collinear
        # vertices, the same outline the frame and reveal were built on.
        clean_group = dataclasses.replace(
            group,
            openings=tuple(
                dataclasses.replace(o, fargs=_flatten(clean_outline[id(o)]))
                for o in group.openings
            ),
        )
        blind_pts = blind_polygons(clean_group)
        blind_prims: list[Primitive] = []
        for k, poly in enumerate(blind_pts):
            blind_name = f"{group.group_id}_blind_{k}"
            names.claim(blind_name)
            blind_prims.append(
                Primitive(blind_mat.name, "polygon", blind_name, (), (), _flatten(poly), "build.prepare")
            )
        blind_prims_by_group[group.group_id] = blind_prims

    scene_prims = [p for p in scene_list if p.is_surface] + generated_surfaces
    materials_prims = [p for p in scene_list if not p.is_surface] + generated_materials

    points_file = project.grid.points_file
    points = read_points(points_file) if points_file is not None else None
    grids = build_grids(spaces, name_to_room, project.grid, case, layout=layout, points=points)
    _check_grid_slug_collisions(grids)

    # -- serialise (nothing touches the disk yet) ------------------------
    blinds_dir = out_dir / "blinds"
    grids_dir = out_dir / "grids"
    scene_file = out_dir / "scene.rad"
    materials_file = out_dir / "materials.rad"
    manifest_path = out_dir / "manifest.json"

    contents: dict[pathlib.Path, bytes] = {
        scene_file: _format_scene(scene_prims).encode("utf-8"),
        materials_file: _format_scene(materials_prims).encode("utf-8"),
    }
    blind_files: dict[str, pathlib.Path] = {}
    for group in groups:
        if group.group_id not in blind_prims_by_group:
            continue  # a diffusing group: no blind file (spec 5.8)
        path = blinds_dir / f"{group.group_id}.rad"
        contents[path] = _format_scene(blind_prims_by_group[group.group_id]).encode("utf-8")
        blind_files[group.group_id] = path
    grid_files: dict[str, pathlib.Path] = {}
    for room_grid in grids:
        path = grids_dir / f"{_slug(room_grid.room)}.pts"
        contents[path] = _format_pts(room_grid.points).encode("utf-8")
        grid_files[room_grid.room] = path

    files_hashes: dict[str, str] = {
        path.relative_to(out_dir).as_posix(): hashlib.sha256(data).hexdigest()
        for path, data in contents.items()
    }

    glazing_manifest: dict[str, dict] = {}
    for record in glazing_records:
        entry = {
            "visible_transmittance": record.visible_transmittance,
            "dirt_factor": record.dirt_factor,
            "net_transmittance": record.net_transmittance,
            "tn": record.transmissivity,
            "tinted": record.tinted,
        }
        if extended:
            entry["class"] = record.tilt_class
            entry["transmission"] = record.transmission
        glazing_manifest[record.modifier] = entry

    groups_manifest: dict[str, dict] = {}
    for group in groups:
        entry = {
            "room": group.room,
            "storey": group.storey,
            "azimuth_bin_deg": group.azimuth_bin_deg,
            "opening_count": len(group.openings),
            "openings": [o.name for o in group.openings],
            "blind_polygon_count": len(blind_prims_by_group.get(group.group_id, ())),
        }
        if extended:
            entry["kind"] = group.kind
            entry["tilt_deg"] = group.tilt_deg
        groups_manifest[group.group_id] = entry

    rooms_manifest = {
        room_grid.room: {
            "area_m2": room_grid.room_area_m2,
            "point_count": int(room_grid.points.shape[0]),
            "work_plane_m": project.grid.work_plane_m,
        }
        for room_grid in grids
    }

    totals = {
        "group_count": len(groups),
        "grid_point_count": sum(int(g.points.shape[0]) for g in grids),
        "regularly_occupied_area_m2": sum(g.room_area_m2 for g in grids),
    }

    # Spec 6.0 WARNING: every grid room with no window group of its own
    # names itself, rather than silently showing no daylight contribution
    # with nothing looking wrong (I5).
    rooms_with_glazing = {group.room for group in groups}
    no_glazing_warnings = tuple(
        f"{room_grid.room}: room has no exterior glazing (no window group); "
        "its sDA/ASE will show no daylight contribution"
        for room_grid in grids
        if room_grid.room not in rooms_with_glazing
    )

    warnings = (
        tuple(project.warnings)
        + report.warnings
        + rooms.warnings
        + materials_warnings
        + area_warnings
        + piece_row_warnings(spaces)
        + group_warnings
        + ceiling_warnings
        + no_glazing_warnings
    )

    manifest = {
        "tool_version": TOOL_VERSION,
        "case": case,
        "scene_units": project.scene.units,
        "standard": project.standard,
        "site": {
            "north_rotation_deg": project.site.north_rotation_deg,
            "north_rotation_source": project.site.north_rotation_source,
        },
        "glazing": glazing_manifest,
        "openings": opening_records,
        "groups": groups_manifest,
        "rooms": rooms_manifest,
        "totals": totals,
        "warnings": list(warnings),
        "files": files_hashes,
    }
    if extended:
        manifest["skylight_well"] = _well_manifest(project.glazing)
    # Only with grid.points_file, so a lattice project keeps its manifest.
    if points is not None:
        manifest["grid_source"] = "points_file"
    # merge-split spec 8: only when a space of this case is split or merged.
    layout_manifest = {
        space.room: _layout_entry(layout.space(space.room))
        for space in selected
        if layout.space(space.room).origin != "detected"
    }
    if layout_manifest:
        manifest["layout"] = layout_manifest

    # I6: a NaN or Infinity that reached this far (none should, since
    # config.py's _as_float already refuses a non-finite project.yaml
    # value) must not be written as bare, invalid JSON -- allow_nan=False
    # turns it into a refusal here instead. Serialised BEFORE any file is
    # written, so that refusal leaves the output folder as it was.
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

    # -- write ---------------------------------------------------------
    # Every check has passed. Each file goes to a temporary name in its own
    # folder and is renamed into place, so no file is ever half-written;
    # the manifest goes last, so a manifest on disk always describes files
    # that are all already there.
    out_dir.mkdir(parents=True, exist_ok=True)
    blinds_dir.mkdir(exist_ok=True)
    grids_dir.mkdir(exist_ok=True)
    for path, data in contents.items():
        _write_atomic(path, data)
    for folder, pattern in ((blinds_dir, "*.rad"), (grids_dir, "*.pts")):
        for existing in folder.glob(pattern):
            if existing not in contents:
                existing.unlink()
    _write_atomic(manifest_path, manifest_bytes)

    # Spec 5.4: the groups keep today's (unframed) opening geometry and
    # differ only in carrying each pane's new modifier.
    prepared_groups = tuple(
        dataclasses.replace(
            group,
            openings=tuple(
                o if pane_modifier.get(o.name, o.modifier) == o.modifier
                else dataclasses.replace(o, modifier=pane_modifier[o.name])
                for o in group.openings
            ),
        )
        for group in groups
    )

    return PreparedModel(
        root=out_dir,
        case=case,
        scene_file=scene_file,
        materials_file=materials_file,
        blind_files=blind_files,
        grid_files=grid_files,
        manifest_path=manifest_path,
        manifest=manifest,
        groups=prepared_groups,
        grids=tuple(grids),
        glass_by_group=glass_by_group,
        effective_roles=effective_roles,
        layout=layout,
    )
