# leed_daylight/engine/coefficients.py
"""Daylight-coefficient matrices per window group, and the lux boundary
(layer 4; spec 6.6 items 2-3, spec 6.8 item 3, spec 6.11).

For each window group, :func:`coefficients` builds two matrices: a
sky-coefficient matrix (``rfluxmtx`` against the annual sky basis, ``full``
ambient parameters) and a direct-sun matrix (``rcontrib -ab 0`` against the
unit-sun scene). A window group is isolated by blacking out every OTHER
group's glazing in a private copy of the scene -- ``scene.rad`` keeps every
exterior opening under its original, shared glass modifier, so a material
override alone cannot isolate one group (R-P34).

Only GROUPED openings are blacked out there, so light that reaches a sensor
through no window group at all -- a skylight labelled ``glazing_interior``,
``ignore`` or ``fixed_shading``, an unglazed gap -- is in every group's own
matrix. :func:`residual_coefficients` traces that light on its own, once per
trace unit (:func:`residual_units`), with EVERY group's openings blacked out
and no blinds, under the pseudo-group id ``RESIDUAL_ID``. Paths that cross
two groups' glass are still lost (P43, accepted).

This module is the ONLY place in the whole tool that Radiance RGB becomes
lux (spec 6.8 item 3): :func:`to_lux` (``matrix @ sky``, per channel, then
``units.rgb_to_lux``) for the sky path, and ``Option1Tracer.direct``'s own
per-sun ``units.rgb_to_lux`` call for the direct-sun path -- both live
here, so nothing above this module ever sees a radiance value.

:class:`Option1Tracer` is the only surface T4.6's Option 1 driver calls: it
wraps :func:`coefficients`, the sky matrix and the sun matrix behind three
methods (``direct``, ``sky``, ``residual``) that return lux, masked to the requested
hours, and trace a closed-blind state only when a schedule actually closes
it (spec 6.6 item 4: the prior job wasted six matrices on groups whose
blinds never closed).
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re
import shlex
from collections.abc import Mapping, Sequence

import numpy as np

from .. import units
from ..build.grid import RoomGrid
from ..build.groups import WindowGroup
from ..build.prepare import PreparedModel
from ..project.config import Performance, Project
from ..scene.primitives import Primitive
from ..scene.reader import quote_word, read_scene
from .aux_files import aux_file_hashes, raypath_dirs
from .fingerprint import delete_sidecar, fingerprint_of, is_current, write_sidecar
from .jobs import checked_below
from .radiance import Radiance, process_count
from .sky import Suns, read_suns

#: The engine's own black-out material, used to isolate one window group by
#: blacking out every other group's glazing (R-P34). Refused by name if it
#: already exists among the prepared materials or surfaces -- silently
#: reusing someone else's material would isolate the wrong openings.
_BLACK_MATERIAL_NAME = "leed_daylight_black"

#: The sky/ground receiver primitive names written into ``sky_receiver.rad``
#: (R-P34 step 4). Held as module constants so every group's receiver file
#: is byte-identical in shape.
_GROUND_MOD = "leed_daylight_ground_glow"
_GROUND_NAME = "leed_daylight_ground"
_SKY_MOD = "leed_daylight_sky_glow"
_SKY_NAME = "leed_daylight_sky"

#: Every name the engine adds to a scene. rfluxmtx(1) requires receiver
#: modifiers to be "unique and not used elsewhere", so each is refused by
#: name if the prepared model already uses it.
_ENGINE_NAMES = (_BLACK_MATERIAL_NAME, _GROUND_MOD, _GROUND_NAME, _SKY_MOD, _SKY_NAME)

#: A ``performance.linked_zones`` key must be a non-negative integer, or a
#: string spelling one in decimal (R-P36).
_DIGITS_RE = re.compile(r"^[0-9]+$")

#: The pseudo-group id of the residual pass (:func:`residual_coefficients`):
#: its matrices are cached under ``engine/groups/@residual/`` and
#: fingerprinted with this id. A real group id is ``slug(room)_<bin>``, and a
#: slug never contains ``@`` (only ``[A-Za-z0-9_-]``), so the two can never
#: collide.
RESIDUAL_ID = "@residual"

#: The sidecar suffix :func:`coefficients` writes the sensor order under,
#: beside each matrix it produces.
_SENSORS_SUFFIX = ".sensors.txt"


class CoefficientError(ValueError):
    """A daylight-coefficient matrix cannot be built, or a matrix file (or
    its sensor order) cannot be read, from these inputs."""


@dataclasses.dataclass(frozen=True)
class GroupMatrices:
    group_id: str
    sky: pathlib.Path  # the group's sky-coefficient matrix
    sun: pathlib.Path  # the group's direct-sun matrix
    sensor_index: tuple[str, ...]  # room name per point row, in write order


# --- R-P36: which grids a group traces against -----------------------------


def _validate_linked_zone_key(key: object) -> int:
    """A storey index: an ``int >= 0``, or a ``str`` of decimal digits."""
    if isinstance(key, bool):
        raise CoefficientError(
            f"performance.linked_zones key {key!r} must be a non-negative "
            "integer storey index, or a string of decimal digits -- not a bool"
        )
    if isinstance(key, int):
        if key < 0:
            raise CoefficientError(
                f"performance.linked_zones key {key!r} must be a non-negative "
                "integer storey index"
            )
        return key
    if isinstance(key, str) and _DIGITS_RE.match(key):
        return int(key)
    raise CoefficientError(
        f"performance.linked_zones key {key!r} must be a non-negative "
        "integer storey index, or a string of decimal digits"
    )


def sensors_in_scope(
    group: WindowGroup, grids: Sequence[RoomGrid], performance: Performance,
) -> list[RoomGrid]:
    """The grids one window group is traced against (R-P36).

    ``trace_scope == "all"`` -> every grid in ``grids``. ``"storey"`` ->
    the grids on ``group.storey``, plus the grids named under that
    storey's ``performance.linked_zones`` key. Every key and every room in
    ``performance.linked_zones`` is validated on every call, whatever
    ``trace_scope`` is: a key that is not a non-negative integer (or a
    digit string), a key naming a storey with no grid in ``grids``, or a
    room not among ``grids`` is a named :class:`CoefficientError`. Any
    other ``trace_scope`` is refused. Grids come back in the order of the
    ``grids`` argument, each at most once.
    """
    grids = list(grids)
    rooms_present = {g.room for g in grids}
    storeys_present = {g.storey for g in grids}

    linked_by_storey: dict[int, tuple[str, ...]] = {}
    for raw_key, rooms in performance.linked_zones.items():
        storey = _validate_linked_zone_key(raw_key)
        if storey not in storeys_present:
            raise CoefficientError(
                f"performance.linked_zones names storey {storey}, which has "
                "no rooms"
            )
        for room in rooms:
            if not isinstance(room, str):
                raise CoefficientError(
                    f"performance.linked_zones storey {storey}: room name "
                    f"must be a string, got {room!r}"
                )
            if room not in rooms_present:
                raise CoefficientError(
                    f"performance.linked_zones storey {storey}: room "
                    f"{room!r} is not among the grids"
                )
        linked_by_storey[storey] = tuple(rooms)

    if performance.trace_scope == "all":
        selected = list(grids)
    elif performance.trace_scope == "storey":
        linked_rooms = set(linked_by_storey.get(group.storey, ()))
        selected = [
            g for g in grids
            if g.storey == group.storey or g.room in linked_rooms
        ]
    else:
        raise CoefficientError(
            "performance.trace_scope must be 'storey' or 'all', got "
            f"{performance.trace_scope!r}"
        )

    seen: set[int] = set()
    deduped: list[RoomGrid] = []
    for g in selected:
        if id(g) not in seen:
            seen.add(id(g))
            deduped.append(g)
    return deduped


def _sensor_index_of(scope: Sequence[RoomGrid]) -> tuple[str, ...]:
    index: list[str] = []
    for grid in scope:
        index.extend([grid.room] * int(grid.points.shape[0]))
    return tuple(index)


def _sensors_sidecar_path(matrix_path: pathlib.Path) -> pathlib.Path:
    return matrix_path.parent / f"{matrix_path.name}{_SENSORS_SUFFIX}"


def _write_sensor_index(matrix_path: pathlib.Path, sensor_index: Sequence[str]) -> None:
    path = _sensors_sidecar_path(matrix_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{room}\n" for room in sensor_index), encoding="utf-8", newline="\n")


# --- R-P34: isolating one window group --------------------------------------


def _blacked_openings(prepared: PreparedModel, group: WindowGroup) -> list[str]:
    """Every OTHER group's opening names, sorted (R-P34 step 2)."""
    return sorted(
        opening.name
        for other in prepared.groups
        if other.group_id != group.group_id
        for opening in other.openings
    )


def _all_group_openings(prepared: PreparedModel) -> list[str]:
    """EVERY group's opening names, sorted: what the residual pass blacks
    out."""
    return sorted(
        opening.name for group in prepared.groups for opening in group.openings
    )


def _primitive_text(
    modifier: str, ptype: str, name: str,
    sargs: Sequence[str], iargs: Sequence[int], fargs: Sequence[float],
) -> str:
    """One Radiance primitive, written by hand (R-P34: never import a
    private ``build/`` formatter). Every word goes through
    :func:`scene.reader.quote_word`, so a quoted word read from the
    prepared model is written back quoted."""
    lines = [
        " ".join(quote_word(w) for w in (modifier, ptype, name)),
        " ".join([str(len(sargs)), *(quote_word(w) for w in sargs)]),
        " ".join([str(len(iargs)), *(str(i) for i in iargs)]),
        " ".join([str(len(fargs)), *(repr(float(f)) for f in fargs)]),
    ]
    return "\n".join(lines) + "\n"


def _existing_names(prepared: PreparedModel) -> tuple[set[str], list[Primitive]]:
    surfaces = read_scene([prepared.scene_file], "m")
    materials = read_scene([prepared.materials_file], "m")
    names = {p.name for p in surfaces} | {p.name for p in materials}
    return names, surfaces


def _write_isolated_rad(
    prepared: PreparedModel, blacked: Sequence[str], work_dir: pathlib.Path,
) -> pathlib.Path:
    """``isolated.rad``: every scene surface unchanged, except the openings
    named in ``blacked`` -- every OTHER group's for a group trace (R-P34
    step 2), every group's for the residual pass."""
    existing_names, surfaces = _existing_names(prepared)
    for name in _ENGINE_NAMES:
        if name in existing_names:
            raise CoefficientError(
                f"{name!r}: a name the engine adds to the scene already exists "
                "among the prepared materials or surfaces; rename it in the "
                "scene, since isolating a window group cannot proceed"
            )
    blacked = set(blacked)

    work_dir.mkdir(parents=True, exist_ok=True)
    isolated_path = work_dir / "isolated.rad"
    parts = [
        _primitive_text("void", "plastic", _BLACK_MATERIAL_NAME, (), (), (0.0, 0.0, 0.0, 0.0, 0.0))
    ]
    for p in surfaces:
        if p.name in blacked:
            p = dataclasses.replace(p, modifier=_BLACK_MATERIAL_NAME)
        parts.append(_primitive_text(p.modifier, p.ptype, p.name, p.sargs, p.iargs, p.fargs))
    isolated_path.write_text("".join(parts), encoding="utf-8", newline="\n")
    return isolated_path


def _build_sky_octree(
    rad: Radiance, prepared: PreparedModel, blacked: Sequence[str],
    work_dir: pathlib.Path, blind_file: pathlib.Path | None,
) -> pathlib.Path:
    isolated = _write_isolated_rad(prepared, blacked, work_dir)
    argv: list[str] = ["oconv", str(prepared.materials_file), str(isolated)]
    if blind_file is not None:
        argv.append(str(blind_file))
    octree = work_dir / "scene.oct"
    rad.run(argv, stdout_path=octree)
    return octree


def _build_sun_octree(
    rad: Radiance, prepared: PreparedModel, blacked: Sequence[str],
    work_dir: pathlib.Path, suns: pathlib.Path,
) -> pathlib.Path:
    # rcontrib collects every modifier in suns.mod; a scene name equal to
    # one would add its own light to that sun's column.
    suns_mod = suns.with_name(suns.name + ".mod")
    sun_modifiers = set(suns_mod.read_text(encoding="utf-8").split())
    clashes = sorted(sun_modifiers & _existing_names(prepared)[0])
    if clashes:
        raise CoefficientError(
            f"{', '.join(repr(c) for c in clashes)}: sun modifier name(s) from "
            f"{suns_mod.name} already exist among the prepared materials or "
            "surfaces; rename them in the scene"
        )
    # Always the open-state isolated scene, whatever blinds_closed is.
    isolated = _write_isolated_rad(prepared, blacked, work_dir)
    argv = ["oconv", str(prepared.materials_file), str(suns), str(isolated)]
    octree = work_dir / "scene_sun.oct"
    rad.run(argv, stdout_path=octree)
    return octree


def _sensors_stdin(points: np.ndarray) -> bytes:
    """One ``rfluxmtx``/``rcontrib`` sensor-ray line per point: ``x y z 0 0
    1`` (position, then the fixed up-facing direction). C1: each component
    is formatted with ``repr(float(x))``, never ``repr`` of the numpy
    scalar itself -- under numpy 2.x, ``f"{x!r}"`` on an ``np.float64``
    writes the text ``np.float64(1.5)``, which no Radiance ASCII reader
    can parse as a number. M9: a ``points`` array whose shape is not
    ``(n, 3)`` (only reachable through a hand-built ``RoomGrid``) is a
    named :class:`CoefficientError`, never a raw unpacking ``ValueError``.
    """
    try:
        points = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise CoefficientError(
            f"sensor points are not a valid array of numbers: {exc}"
        ) from None
    if points.ndim != 2 or points.shape[1] != 3:
        raise CoefficientError(
            f"sensor points must have shape (n, 3), got {points.shape}"
        )
    lines = [f"{float(x)!r} {float(y)!r} {float(z)!r} 0 0 1" for x, y, z in points]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("ascii")


def _write_sky_receiver(work_dir: pathlib.Path, mf: int) -> pathlib.Path:
    path = work_dir / "sky_receiver.rad"
    text = (
        "#@rfluxmtx h=u\n"
        f"void glow {_GROUND_MOD} 0 0 4 1 1 1 0\n"
        f"{_GROUND_MOD} source {_GROUND_NAME} 0 0 4 0 0 -1 180\n"
        f"#@rfluxmtx h=r{mf} u=+Y\n"
        f"void glow {_SKY_MOD} 0 0 4 1 1 1 0\n"
        f"{_SKY_MOD} source {_SKY_NAME} 0 0 4 0 0 1 180\n"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _check_processes(processes: object) -> int:
    """``processes`` must be an ``int >= 1`` (never a bool)."""
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise CoefficientError(
            f"processes must be an integer of at least 1, got {processes!r}"
        )
    return processes


def _processes_args(processes: int, *, platform: str | None = None) -> list[str]:
    """``-n N`` for ``rfluxmtx``/``rcontrib`` when more than one process is
    asked for and the system allows it; nothing at 1, so a single-process
    command line is unchanged. ``-n`` only splits the work across processes
    -- the matrix written is the same -- so it is never part of a fingerprint."""
    processes = process_count(processes, platform=platform)
    return ["-n", str(processes)] if processes > 1 else []


def _run_sky_coefficients(
    rad: Radiance, project: Project, octree: pathlib.Path,
    points: np.ndarray, out: pathlib.Path, processes: int = 1,
) -> None:
    receiver = _write_sky_receiver(out.parent, project.sky.MF)
    argv: list[str] = [
        "rfluxmtx", "-I+", "-y", str(points.shape[0]), "-faf",
        *_processes_args(processes),
        *shlex.split(project.radiance.ambient_full),
        *shlex.split(project.radiance.direct),
        "-", str(receiver), "-i", str(octree),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    rad.run(argv, stdin=_sensors_stdin(points), stdout_path=out)


def _run_sun_coefficients(
    rad: Radiance, project: Project, octree: pathlib.Path,
    points: np.ndarray, suns_mod: pathlib.Path, out: pathlib.Path,
    processes: int = 1,
) -> None:
    argv: list[str] = [
        "rcontrib", "-I+", "-y", str(points.shape[0]),
        *_processes_args(processes),
        *shlex.split(project.radiance.direct),
        "-ab", "0",  # after the user options, so it always wins
        "-faf", "-M", str(suns_mod), str(octree),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    rad.run(argv, stdin=_sensors_stdin(points), stdout_path=out)


def _scope_key(rooms_in_scope: Sequence[str]) -> str:
    """A short, stable key for one trace's sensor scope (I1): two calls
    whose ``rooms_in_scope`` differ (e.g. ``storey`` vs ``all``) always get
    different keys, and the same ordered rooms always get the same key --
    so a work folder built for one scope can never be silently reused (or
    overwritten) by another."""
    digest = hashlib.sha256("\n".join(rooms_in_scope).encode("utf-8")).hexdigest()
    return digest[:16]


def _require_group_in_prepared(prepared: PreparedModel, group: WindowGroup) -> None:
    ids = {g.group_id for g in prepared.groups}
    if group.group_id not in ids:
        raise CoefficientError(
            f"group {group.group_id!r} is not one of prepared.groups "
            f"({sorted(ids)!r})"
        )


def coefficients(
    rad: Radiance, prepared: PreparedModel, group: WindowGroup,
    grids: Sequence[RoomGrid], project: Project, blinds_closed: bool,
    *, suns: pathlib.Path, processes: int = 1,
) -> GroupMatrices:
    """Build (or reuse, if current) one window group's sky and direct-sun
    coefficient matrices (R-P34, R-P31).

    ``processes`` (an ``int >= 1``) is passed to ``rfluxmtx``/``rcontrib``
    as ``-n``; it changes speed only, so it is not fingerprinted.

    Refuses by name: ``group`` not one of ``prepared.groups``; an empty
    sensor scope (no points to trace); ``blinds_closed=True`` for a group
    with no entry in ``prepared.blind_files``; ``project.sky.basis`` other
    than ``"Reinhart"``.
    """
    _require_group_in_prepared(prepared, group)
    processes = _check_processes(processes)

    scope = sensors_in_scope(group, grids, project.performance)
    n_points = sum(int(g.points.shape[0]) for g in scope)
    if not scope or n_points == 0:
        raise CoefficientError(
            f"{group.group_id}: sensor scope is empty; there are no sensor "
            "points to trace"
        )

    if blinds_closed and group.group_id not in prepared.blind_files:
        raise CoefficientError(
            f"{group.group_id}: blinds_closed=True but this group has no "
            "entry in prepared.blind_files"
        )

    if project.sky.basis != "Reinhart":
        raise CoefficientError(
            f"project.sky.basis must be 'Reinhart', got {project.sky.basis!r}"
        )

    blind_file = prepared.blind_files[group.group_id] if blinds_closed else None
    return _trace_unit(
        rad, prepared, project, unit_id=group.group_id, scope=scope,
        blacked=_blacked_openings(prepared, group), blind_file=blind_file,
        suns=suns, processes=processes,
    )


def residual_units(
    grids: Sequence[RoomGrid], performance: Performance,
) -> list[list[RoomGrid]]:
    """The residual pass's trace units: every grid exactly once. One unit
    per storey (in first-seen order) under ``trace_scope: storey``; one
    unit of every grid under ``all``. Empty grids are left out."""
    grids = [g for g in grids if int(g.points.shape[0]) > 0]
    if performance.trace_scope == "all":
        return [grids] if grids else []
    if performance.trace_scope == "storey":
        by_storey: dict[int, list[RoomGrid]] = {}
        for g in grids:
            by_storey.setdefault(g.storey, []).append(g)
        return list(by_storey.values())
    raise CoefficientError(
        "performance.trace_scope must be 'storey' or 'all', got "
        f"{performance.trace_scope!r}"
    )


def residual_coefficients(
    rad: Radiance, prepared: PreparedModel, unit: Sequence[RoomGrid],
    project: Project, *, suns: pathlib.Path, processes: int = 1,
) -> GroupMatrices:
    """Build (or reuse, if current) the RESIDUAL sky and direct-sun
    matrices for one trace unit (a list of grids, from
    :func:`residual_units`): the scene with EVERY window group's openings
    blacked out and no blinds.

    Whatever light this still delivers reaches the sensors through
    surfaces that belong to no window group -- a skylight labelled
    ``glazing_interior``/``ignore``/``fixed_shading``, an unglazed gap.
    Each group's own isolated trace (R-P34) sees that light too, since only
    grouped openings are blacked out there, so a room that n group traces
    cover holds n copies of it: the Option 1 driver keeps the sum of the
    traces and subtracts n - 1 residuals (none for n = 1), and takes the
    residual direct sun out of each group's own direct lux before the
    blind trigger.
    Cached and fingerprinted like a group, under the pseudo-group id
    ``RESIDUAL_ID``, which no real group id can take.
    """
    processes = _check_processes(processes)
    unit = list(unit)
    if not unit or sum(int(g.points.shape[0]) for g in unit) == 0:
        raise CoefficientError(
            f"{RESIDUAL_ID}: trace unit is empty; there are no sensor points to trace"
        )
    if project.sky.basis != "Reinhart":
        raise CoefficientError(
            f"project.sky.basis must be 'Reinhart', got {project.sky.basis!r}"
        )
    return _trace_unit(
        rad, prepared, project, unit_id=RESIDUAL_ID, scope=unit,
        blacked=_all_group_openings(prepared), blind_file=None,
        suns=suns, processes=processes,
    )


def _trace_unit(
    rad: Radiance, prepared: PreparedModel, project: Project, *,
    unit_id: str, scope: Sequence[RoomGrid], blacked: Sequence[str],
    blind_file: pathlib.Path | None, suns: pathlib.Path, processes: int,
) -> GroupMatrices:
    """The sky and direct-sun matrices for one trace unit: ``scope``'s
    sensor points, with ``blacked`` openings blacked out, and (for the sky
    matrix only) ``blind_file`` added when not None. Cached under
    ``engine/groups/<unit_id>/<scope key>/``."""
    blinds_closed = blind_file is not None
    n_points = sum(int(g.points.shape[0]) for g in scope)
    sensor_index = _sensor_index_of(scope)
    try:
        points = np.concatenate([g.points for g in scope], axis=0)
    except ValueError as exc:
        raise CoefficientError(
            f"{unit_id}: could not concatenate sensor points from "
            f"the grids in scope: {exc}"
        ) from None
    rooms_in_scope = [g.room for g in scope]
    blacked = list(blacked)

    # I1: keyed on the ORDERED rooms in scope, not just the group id -- a
    # "storey" scope and an "all" scope on the same prepared model trace
    # different sensor points and must never share (or stomp on) a work
    # folder. Without this, an acceptance session that runs both scopes
    # (spec 6.11's proof obligation) would overwrite one scope's matrices
    # with the other's and re-trace it every time.
    scope_key = _scope_key(rooms_in_scope)
    group_dir = prepared.root / "engine" / "groups" / unit_id / scope_key
    open_dir = group_dir / "open"
    state_dir = (group_dir / "closed") if blinds_closed else open_dir
    # A link anywhere in the cache tree would send the deletes and rebuilds
    # below outside the project; the driver has checked down to the tree's
    # root (re-review OOS-1 and its residual).
    for folder in dict.fromkeys((state_dir, open_dir)):
        checked_below(prepared.root, folder)

    # The files the materials name (.cal, .dat, pictures) are reached through
    # RAYPATH, so they are fingerprinted by content here.
    aux_files = aux_file_hashes(
        read_scene([prepared.materials_file], "m"), raypath_dirs(project, prepared.root),
    )

    # -- sky coefficients ----------------------------------------------
    sky_out = state_dir / "sky.mtx"
    sky_fp = fingerprint_of(
        {"manifest": prepared.manifest_path},
        {
            "command": "coefficients_sky",
            "group_id": unit_id,
            "blind_state": "closed" if blinds_closed else "open",
            "rooms_in_scope": rooms_in_scope,
            "sensor_count": n_points,
            "ambient_full": project.radiance.ambient_full,
            "direct": project.radiance.direct,
            "MF": project.sky.MF,
            "blacked_openings": blacked,
            "aux_files": aux_files,
            "radiance_version": rad.version,
        },
    )
    if not (sky_out.exists() and is_current(sky_out, sky_fp)):
        delete_sidecar(sky_out)  # M1: before the rebuild, not after
        octree = _build_sky_octree(rad, prepared, blacked, state_dir, blind_file)
        _run_sky_coefficients(rad, project, octree, points, sky_out, processes)
        write_sidecar(sky_out, sky_fp)
    _write_sensor_index(sky_out, sensor_index)

    # -- direct-sun coefficients (always blinds open) --------------------
    suns = pathlib.Path(suns)
    suns_mod = suns.with_name(suns.name + ".mod")
    sun_out = open_dir / "sun.mtx"
    sun_fp = fingerprint_of(
        {"manifest": prepared.manifest_path, "suns": suns, "suns_mod": suns_mod},
        {
            "command": "coefficients_sun",
            "group_id": unit_id,
            "blind_state": "open",
            "rooms_in_scope": rooms_in_scope,
            "sensor_count": n_points,
            "ambient_full": project.radiance.ambient_full,
            "direct": project.radiance.direct,
            "MF": project.sky.MF,
            "blacked_openings": blacked,
            "aux_files": aux_files,
            "radiance_version": rad.version,
        },
    )
    if not (sun_out.exists() and is_current(sun_out, sun_fp)):
        delete_sidecar(sun_out)  # M1: before the rebuild, not after
        octree = _build_sun_octree(rad, prepared, blacked, open_dir, suns)
        _run_sun_coefficients(rad, project, octree, points, suns_mod, sun_out, processes)
        write_sidecar(sun_out, sun_fp)
    _write_sensor_index(sun_out, sensor_index)

    return GroupMatrices(group_id=unit_id, sky=sky_out, sun=sun_out, sensor_index=sensor_index)


# --- R-P35: reading a Radiance matrix file back ------------------------------


def _read_header_lines(fh, path: pathlib.Path) -> list[bytes]:
    first = fh.readline()
    if first.strip() != b"#?RADIANCE":
        raise CoefficientError(
            f"matrix {path}: first header line must be '#?RADIANCE', got "
            f"{first.strip()!r}"
        )
    lines: list[bytes] = []
    while True:
        line = fh.readline()
        if line == b"":
            raise CoefficientError(
                f"matrix {path}: header never ends (no blank line found)"
            )
        if line.strip() == b"":
            break
        lines.append(line)
    return lines


#: A header line is a variable only when it starts with a plain
#: identifier followed by '=' (getinfo(1)/rcontrib(1)/gendaymtx(1)): every
#: RADIANCE file's header also carries free-text lines with no such shape
#: -- most commonly the producing command line (e.g. ``gendaymtx -m 4 -r
#: 0.0 -of -c 1 1 1 site.wea``), which rcontrib also copies in from the
#: octree it reads (C2). Those lines are informational and are skipped,
#: never refused; only a missing/malformed REQUIRED key further down still
#: refuses.
_HEADER_VARIABLE_RE = re.compile(rb"^[A-Za-z_][A-Za-z0-9_]*=")


def _parse_header_fields(lines: list[bytes], path: pathlib.Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if not _HEADER_VARIABLE_RE.match(line):
            continue
        key, _, value = line.partition(b"=")
        fields[key.decode("ascii", "replace")] = value.decode("ascii", "replace").strip()
    return fields


def _header_only(path: pathlib.Path) -> dict[str, str]:
    path = pathlib.Path(path)
    try:
        with open(path, "rb") as fh:
            lines = _read_header_lines(fh, path)
    except OSError as exc:
        raise CoefficientError(f"matrix {path}: could not be read: {exc}") from exc
    return _parse_header_fields(lines, path)


def _int_header(fields: Mapping[str, str], key: str, path: pathlib.Path) -> int:
    if key not in fields:
        raise CoefficientError(f"matrix {path}: missing header key {key!r}")
    try:
        value = int(fields[key])
    except ValueError:
        raise CoefficientError(
            f"matrix {path}: header key {key!r} is not an integer: "
            f"{fields[key]!r}"
        ) from None
    if value < 0:
        raise CoefficientError(
            f"matrix {path}: header key {key!r} must be non-negative, got {value}"
        )
    return value


#: How many values :func:`read_matrix` converts at a time from a binary
#: matrix file. Bounds the reader's working memory beyond the result itself
#: (about 12 MB of float64 at this size).
_READ_CHUNK_VALUES = 1 << 20


def read_matrix(path: pathlib.Path, dtype=np.float64) -> np.ndarray:
    """Read a Radiance matrix file back into an ``(rows, cols, 3)`` array
    of ``dtype`` (``float64``, the default, or ``float32``) (R-P35).

    The header is an ASCII ``#?RADIANCE`` line, then ``KEY=value`` lines
    (``NROWS``, ``NCOLS``, ``NCOMP=3``, ``FORMAT=ascii|float|double``,
    optionally ``BYTEORDER`` or ``BigEndian=0|1``), ended by a blank line; the data follows,
    row by row, each column's three components together. ``ascii``,
    ``float`` (4-byte) and ``double`` (8-byte) are supported;
    ``BigEndian=1`` (what rcontrib and gendaymtx write) and ``BYTEORDER``
    are honoured when present (little-endian otherwise).
    Refuses by name: a missing or unreadable file, a missing or malformed
    header key, ``NCOMP != 3``, a data size that disagrees with the
    header, an unsupported ``FORMAT``, or NaN/Inf anywhere in the data.

    Memory: binary data is read straight from the file into the result,
    ``_READ_CHUNK_VALUES`` values at a time, so the peak is the result plus
    one chunk -- never the raw bytes and a full-size converted copy
    together. ``float32`` is exact for a ``FORMAT=float`` file (what
    ``rcontrib -faf`` writes) and halves the result; a caller that sums lux
    converts to float64 per slice before summing.
    """
    path = pathlib.Path(path)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
        raise CoefficientError(
            f"matrix {path}: dtype must be float64 or float32, got {dtype}"
        )
    try:
        with open(path, "rb") as fh:
            header_lines = _read_header_lines(fh, path)
            fields = _parse_header_fields(header_lines, path)
            nrows = _int_header(fields, "NROWS", path)
            ncols = _int_header(fields, "NCOLS", path)
            ncomp = _int_header(fields, "NCOMP", path)
            if ncomp != 3:
                raise CoefficientError(f"matrix {path}: NCOMP must be 3, got {ncomp}")
            if "FORMAT" not in fields:
                raise CoefficientError(f"matrix {path}: missing header key 'FORMAT'")
            fmt = fields["FORMAT"]
            big_endian = (
                fields.get("BigEndian", "").strip() == "1"
                or "big" in fields.get("BYTEORDER", "").lower()
            )
            n_values = nrows * ncols * ncomp

            if fmt == "ascii":
                values = _read_ascii_values(fh.read(), n_values, path).astype(dtype)
            elif fmt in ("float", "double"):
                values = _read_binary_values(fh, fmt, big_endian, n_values, dtype, path)
            else:
                raise CoefficientError(
                    f"matrix {path}: unsupported FORMAT {fmt!r}; expected ascii, "
                    "float or double"
                )
    except OSError as exc:
        raise CoefficientError(f"matrix {path}: could not be read: {exc}") from exc

    return values.reshape(nrows, ncols, ncomp)


def _read_ascii_values(data_bytes: bytes, n_values: int, path: pathlib.Path) -> np.ndarray:
    tokens = data_bytes.split()
    if len(tokens) != n_values:
        raise CoefficientError(
            f"matrix {path}: expected {n_values} ascii values "
            f"(NROWS x NCOLS x NCOMP), found {len(tokens)}"
        )
    try:
        values = np.array([float(t) for t in tokens], dtype=np.float64)
    except ValueError as exc:
        raise CoefficientError(
            f"matrix {path}: non-numeric value in ascii data: {exc}"
        ) from None
    if not np.all(np.isfinite(values)):
        raise CoefficientError(f"matrix {path}: data contains NaN or Inf")
    return values


def _read_binary_values(
    fh, fmt: str, big_endian: bool, n_values: int, dtype: np.dtype, path: pathlib.Path,
) -> np.ndarray:
    """The binary payload, converted chunk by chunk into a ``dtype`` array."""
    itemsize = 4 if fmt == "float" else 8
    source = np.dtype((">" if big_endian else "<") + ("f4" if fmt == "float" else "f8"))
    expected_bytes = n_values * itemsize

    data_start = fh.tell()
    fh.seek(0, 2)
    found_bytes = fh.tell() - data_start
    if found_bytes != expected_bytes:
        trailing_ok = False
        if found_bytes == expected_bytes + 1:
            fh.seek(data_start + expected_bytes)
            trailing_ok = fh.read(1) in (b"\n", b"\r")
        if not trailing_ok:
            raise CoefficientError(
                f"matrix {path}: expected {expected_bytes} bytes of "
                f"{fmt} data (NROWS x NCOLS x NCOMP = {n_values} values), "
                f"found {found_bytes}"
            )
    fh.seek(data_start)

    values = np.empty(n_values, dtype=dtype)
    done = 0
    while done < n_values:
        count = min(_READ_CHUNK_VALUES, n_values - done)
        chunk = np.frombuffer(fh.read(count * itemsize), dtype=source)
        if chunk.shape[0] != count:
            raise CoefficientError(
                f"matrix {path}: data ended early, after {done + chunk.shape[0]} "
                f"of {n_values} values"
            )
        values[done:done + count] = chunk
        # Checked after conversion, so a double too large for float32 is
        # refused too, never silently turned into Inf.
        if not np.all(np.isfinite(values[done:done + count])):
            raise CoefficientError(f"matrix {path}: data contains NaN or Inf")
        done += count
    return values


def read_matrix_compact(path: pathlib.Path) -> np.ndarray:
    """:func:`read_matrix` at the file's own precision: ``float32`` for a
    ``FORMAT=float`` file (exact, half the memory of float64), ``float64``
    otherwise. Every lux sum made from it converts to float64 per slice
    first, so the lux is bit-identical to a float64 read."""
    fields = _header_only(path)
    dtype = np.float32 if fields.get("FORMAT") == "float" else np.float64
    return read_matrix(path, dtype=dtype)


# --- the lux boundary ---------------------------------------------------


def to_lux(matrix: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """The sky-matrix half of where Radiance RGB becomes lux (spec 6.8
    item 3; the other half is ``Option1Tracer.direct``'s own per-sun
    ``units.rgb_to_lux`` call, in this same module).

    ``matrix`` is ``(points, patches, 3)``, ``sky`` is
    ``(patches, hours, 3)``; per channel, ``matrix[:, :, c] @ sky[:, :,
    c]``, then :func:`units.rgb_to_lux`. Returns ``(points, hours)``.
    Refuses by name a wrong ndim (either array), a last axis that is not
    3 (either array), ``matrix.shape[1] != sky.shape[0]`` (showing both
    shapes), or NaN/Inf anywhere in either array.

    A float32 ``matrix`` (from :func:`read_matrix_compact`) is not copied
    whole to float64: each channel is converted on its own just before its
    product, which gives the same float64 result with a third of the copy.
    """
    try:
        matrix = np.asarray(matrix)
        if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            matrix = np.asarray(matrix, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise CoefficientError(
            f"to_lux: matrix is not a valid array of numbers: {exc}"
        ) from None
    try:
        sky = np.asarray(sky, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise CoefficientError(
            f"to_lux: sky is not a valid array of numbers: {exc}"
        ) from None

    if matrix.ndim != 3:
        raise CoefficientError(
            f"to_lux: matrix must be 3-D (points, patches, 3), got shape "
            f"{matrix.shape}"
        )
    if sky.ndim != 3:
        raise CoefficientError(
            f"to_lux: sky must be 3-D (patches, hours, 3), got shape {sky.shape}"
        )
    if matrix.shape[-1] != 3:
        raise CoefficientError(
            f"to_lux: matrix's last axis must be 3 (RGB), got shape {matrix.shape}"
        )
    if sky.shape[-1] != 3:
        raise CoefficientError(
            f"to_lux: sky's last axis must be 3 (RGB), got shape {sky.shape}"
        )
    if matrix.shape[1] != sky.shape[0]:
        raise CoefficientError(
            f"to_lux: matrix.shape[1] ({matrix.shape[1]}) must equal "
            f"sky.shape[0] ({sky.shape[0]}); shapes are matrix {matrix.shape}, "
            f"sky {sky.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise CoefficientError("to_lux: matrix contains NaN or Inf")
    if not np.all(np.isfinite(sky)):
        raise CoefficientError("to_lux: sky contains NaN or Inf")

    n_points, n_patches, _ = matrix.shape
    n_hours = sky.shape[1]
    rgb_result = np.empty((n_points, n_hours, 3), dtype=np.float64)
    for c in range(3):
        rgb_result[:, :, c] = matrix[:, :, c].astype(np.float64, copy=False) @ sky[:, :, c]
    return units.rgb_to_lux(rgb_result)


# --- Option1Tracer: the only surface T4.6 calls ------------------------------


class Option1Tracer:
    """The Option 1 driver's only entry point into this module.

    Wraps :func:`coefficients`, the annual sky matrix and the annual sun
    matrix behind three lux-only methods (``direct``, ``sky``,
    ``residual``). The sky matrix is read once and cached; a closed-blind
    trace only happens when ``open_schedule`` actually has a closed hour.
    """

    def __init__(
        self, rad: Radiance, project: Project, prepared: PreparedModel,
        *, sky: pathlib.Path, suns: pathlib.Path, processes: int = 1,
    ) -> None:
        self._processes = _check_processes(processes)
        self._rad = rad
        self._project = project
        self._prepared = prepared
        self._sky_path = pathlib.Path(sky)
        self._suns_path = pathlib.Path(suns)
        self._n_hours: int | None = None
        self._sky_array: np.ndarray | None = None
        self._sky_masked: np.ndarray | None = None
        self._sky_masked_key: bytes | None = None
        self._suns: Suns | None = None

    @property
    def n_hours(self) -> int:
        """The sky matrix's ``NCOLS``, from its header only."""
        if self._n_hours is None:
            fields = _header_only(self._sky_path)
            self._n_hours = _int_header(fields, "NCOLS", self._sky_path)
        return self._n_hours

    def _sky_array_full(self) -> np.ndarray:
        """The full annual sky matrix, read once and cached as float32
        (I2): halves its resident memory next to the float64 it was read
        as. ``to_lux`` promotes back to float64 before the actual dot
        product, so the LUX RESULT is still computed in float64 -- only
        the cached matrix itself is narrower.
        """
        if self._sky_array is None:
            # Read straight into float32 (the same rounding as a float64
            # read then astype), never holding a float64 copy.
            self._sky_array = read_matrix(self._sky_path, dtype=np.float32)
        return self._sky_array

    def _sky_array_masked(self, hours: np.ndarray) -> np.ndarray:
        """``self._sky_array_full()[:, hours, :]``, masked and cached ONCE
        per distinct ``hours`` mask (I2): every window group in one
        ``run_option1`` call shares the SAME occupied-hour mask, so
        without this cache every one of them would copy its own
        boolean-indexed slice of the (potentially hundreds-of-MB) annual
        sky matrix."""
        key = np.asarray(hours, dtype=bool).tobytes()
        if self._sky_masked is None or self._sky_masked_key != key:
            self._sky_masked = self._sky_array_full()[:, hours, :]
            self._sky_masked_key = key
        return self._sky_masked

    def _suns_data(self) -> Suns:
        if self._suns is None:
            self._suns = read_suns(self._suns_path)
        return self._suns

    def _validated_hours(self, hours) -> np.ndarray:
        try:
            hours = np.asarray(hours)
        except (ValueError, TypeError) as exc:
            raise CoefficientError(f"hours is not a valid array: {exc}") from None
        if hours.ndim != 1:
            raise CoefficientError(
                f"hours must be a 1-D bool mask, got shape {hours.shape}"
            )
        if hours.shape[0] != self.n_hours:
            raise CoefficientError(
                f"hours mask has length {hours.shape[0]}, does not match "
                f"n_hours {self.n_hours}"
            )
        return hours.astype(bool)

    def _scope_and_index(self, group: WindowGroup) -> tuple[list[RoomGrid], tuple[str, ...], int]:
        scope = sensors_in_scope(group, self._prepared.grids, self._project.performance)
        sensor_index = _sensor_index_of(scope)
        n_points = sum(int(g.points.shape[0]) for g in scope)
        return scope, sensor_index, n_points

    def direct(
        self, group: WindowGroup, hours: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """The group's direct-sun lux, blinds open, masked to ``hours``."""
        hours = self._validated_hours(hours)
        n_selected = int(hours.sum())

        scope, sensor_index, n_points = self._scope_and_index(group)
        if not scope or n_points == 0:
            return (), np.zeros((0, n_selected), dtype=np.float64)

        matrices = coefficients(
            self._rad, self._prepared, group, self._prepared.grids,
            self._project, False, suns=self._suns_path, processes=self._processes,
        )
        return matrices.sensor_index, self._sun_lux(
            matrices, group.group_id, n_points, hours,
        )

    def _sun_lux(
        self, matrices: GroupMatrices, label: str, n_points: int, hours: np.ndarray,
    ) -> np.ndarray:
        """Direct-sun lux from one trace unit's sun matrix, masked to
        ``hours``: ``(n_points, hours.sum())``."""
        n_selected = int(hours.sum())
        # (points, suns, 3), float32 for a -faf file: a 20,000-point sun
        # matrix is the largest array this tool reads. Each per-sun slice is
        # promoted to float64 by the multiply below, so the lux is unchanged.
        coef = read_matrix_compact(matrices.sun)
        suns = self._suns_data()

        # M2: a bad sun matrix file must be refused by name here, not
        # surface as a raw numpy broadcast/index error three lines below.
        if coef.shape[0] != n_points:
            raise CoefficientError(
                f"{label}: sun matrix {matrices.sun} has "
                f"NROWS={coef.shape[0]}, expected {n_points} (the group's "
                "own sensor count)"
            )
        if coef.shape[1] != len(suns.modifiers):
            raise CoefficientError(
                f"{label}: sun matrix {matrices.sun} has "
                f"NCOLS={coef.shape[1]}, expected {len(suns.modifiers)} "
                "(len(suns.modifiers))"
            )

        hour_to_col = -np.ones(self.n_hours, dtype=np.int64)
        hour_to_col[hours] = np.arange(n_selected)

        result = np.zeros((n_points, n_selected), dtype=np.float64)
        for s in range(len(suns.modifiers)):
            hour = int(suns.hours[s])
            if hour < 0 or hour >= self.n_hours:
                raise CoefficientError(
                    f"sun {suns.modifiers[s]!r} has hour index {hour}, out "
                    f"of range [0, {self.n_hours})"
                )
            col = hour_to_col[hour]
            if col < 0:
                continue
            contribution = units.rgb_to_lux(coef[:, s, :] * suns.rgb[s])
            result[:, col] += contribution
        return result

    def residual(self, hours: np.ndarray) -> "_ResidualUnits":
        """The residual pass (:func:`residual_coefficients`), one entry per
        trace unit from :func:`residual_units`: ``(sensor_index, sky_lux,
        direct_lux)``, both masked to ``hours``, with every window group's
        openings blacked out and no blinds. Together the units cover every
        grid exactly once. Traced lazily: each unit is traced (or reused)
        as it is iterated, so only one unit's lux is alive at a time."""
        hours = self._validated_hours(hours)
        return _ResidualUnits(self, hours)

    def _residual_unit(
        self, unit: Sequence[RoomGrid], hours: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        n_points = sum(int(g.points.shape[0]) for g in unit)
        matrices = residual_coefficients(
            self._rad, self._prepared, unit, self._project,
            suns=self._suns_path, processes=self._processes,
        )
        sky_lux = to_lux(read_matrix_compact(matrices.sky), self._sky_array_masked(hours))
        direct_lux = self._sun_lux(matrices, RESIDUAL_ID, n_points, hours)
        return matrices.sensor_index, sky_lux, direct_lux


    def sky(
        self, group: WindowGroup, hours: np.ndarray, open_schedule: np.ndarray,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
        """The group's blinds-open annual lux, masked to ``hours``, and --
        only when ``open_schedule`` has a closed hour -- the blinds-closed
        annual lux; otherwise ``None`` without ever tracing the closed
        state."""
        hours = self._validated_hours(hours)
        n_selected = int(hours.sum())

        try:
            open_schedule = np.asarray(open_schedule)
        except (ValueError, TypeError) as exc:
            raise CoefficientError(f"open_schedule is not a valid array: {exc}") from None
        if open_schedule.ndim != 1 or open_schedule.shape[0] != n_selected:
            raise CoefficientError(
                "open_schedule must be a 1-D bool array of length "
                f"hours.sum() ({n_selected}), got shape {open_schedule.shape}"
            )
        open_schedule = open_schedule.astype(bool)

        scope, sensor_index, n_points = self._scope_and_index(group)
        if not scope or n_points == 0:
            return (), np.zeros((0, n_selected), dtype=np.float64), None

        sky_masked = self._sky_array_masked(hours)  # (patches, n_selected, 3)

        open_matrices = coefficients(
            self._rad, self._prepared, group, self._prepared.grids,
            self._project, False, suns=self._suns_path, processes=self._processes,
        )
        open_coef = read_matrix_compact(open_matrices.sky)
        open_lux = to_lux(open_coef, sky_masked)

        closed_lux: np.ndarray | None = None
        if not np.all(open_schedule):
            closed_matrices = coefficients(
                self._rad, self._prepared, group, self._prepared.grids,
                self._project, True, suns=self._suns_path, processes=self._processes,
            )
            closed_coef = read_matrix_compact(closed_matrices.sky)
            closed_lux = to_lux(closed_coef, sky_masked)

        return open_matrices.sensor_index, open_lux, closed_lux


class _ResidualUnits:
    """What :meth:`Option1Tracer.residual` returns: iterating it traces
    (or reuses) one residual unit at a time."""

    def __init__(self, tracer: Option1Tracer, hours: np.ndarray) -> None:
        self._tracer = tracer
        self._hours = hours
        self._units = residual_units(tracer._prepared.grids, tracer._project.performance)

    def __len__(self) -> int:
        return len(self._units)

    def __iter__(self):
        for unit in self._units:
            yield self._tracer._residual_unit(unit, self._hours)
