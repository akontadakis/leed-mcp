"""Quality Views ray marching (layer 4; QV spec 6.4).

Every view ray is traced through the black views model with
``rtrace -n <processes> -ab 0 -oLpNsm``: the length to the first hit, the hit point, the
unperturbed surface normal, the surface name and the modifier name. A ray continues past a hit, from just
beyond the hit point in the same direction, while:

* it has not yet passed exterior glass, and the hit is see-through
  (``pass_through``); an ``exterior`` hit is the exit;
* it has passed exterior glass, and the hit is closer than
  ``skip_within_m`` beyond the exit, or is see-through but not exterior
  glass, or is exterior glass within LAYER_MERGE_M of the exit (the other
  face of the same glazing).

Anything else stops the ray: that hit is what the ray reaches or sees.
This module returns names and hit points only. It never decides a view
type (metrics.views does), and it never returns a Radiance colour.

This module is the one documented exception to the parent spec's
"the engine returns arrays in lux" rule: a view ray has no lux.
"""
from __future__ import annotations

import io
import pathlib
from dataclasses import dataclass

import numpy as np

from .fingerprint import delete_sidecar, fingerprint_of, is_current, write_sidecar
from .radiance import Radiance, RadianceError, process_count

#: Per-segment modifier index for a ray that hit nothing / a segment never traced.
MISS = -1
UNUSED = -2
#: QV spec 6.4: glass hits this close are one layer. Same value as metrics.views.
LAYER_MERGE_M = 0.10
#: Two glass faces are one layer only if their normals are this close to
#: parallel. Same value as metrics.views.
LAYER_PARALLEL_COS = 0.999
#: How far past a hit point the next segment starts, in metres.
NUDGE_M = 1e-3
#: rtrace reports a ray that hits nothing with a huge length (FHUGE, 1e10).
#: UNCONFIRMED against real Radiance until the smoke run (plan Task 12).
MISS_LENGTH_MIN = 1e9
#: Rays per rtrace call.
CHUNK_RAYS = 500_000

_RTRACE_ARGS = ("-h", "-faa", "-ab", "0", "-av", "0", "0", "0", "-oLpNsm")


class ViewTraceError(RadianceError):
    """View rays could not be traced, or rtrace's answer cannot be read."""


@dataclass(frozen=True)
class RaySegments:
    modifiers: tuple[str, ...]
    surfaces: tuple[str, ...]
    #: (n, S) index into ``modifiers``; MISS or UNUSED otherwise.
    modifier: np.ndarray
    #: (n, S) index into ``surfaces``; -1 for a miss or an unused segment.
    surface: np.ndarray
    #: (n, S, 3) hit points in metres; NaN for a miss or an unused segment.
    hit: np.ndarray
    #: (n, S, 3) surface normals at the hits (rtrace's unperturbed N, sign
    #: as the surface was modelled); NaN for a miss or an unused segment.
    #: None when the tracer does not report normals: layers are then judged
    #: by the distance along the ray alone.
    normal: np.ndarray | None = None


def build_view_octree(rad: Radiance, scene, materials, work_dir) -> pathlib.Path:
    """``oconv materials scene``, rebuilt only when either file changed."""
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    octree = work_dir / "views.oct"
    fp = fingerprint_of(
        {"scene": pathlib.Path(scene), "materials": pathlib.Path(materials)},
        {"kind": "views_octree"},
    )
    if is_current(octree, fp):
        return octree
    delete_sidecar(octree)
    rad.run(["oconv", str(materials), str(scene)], stdout_path=octree)
    write_sidecar(octree, fp)
    return octree


def _rays_stdin(origins: np.ndarray, directions: np.ndarray) -> bytes:
    buffer = io.StringIO()
    np.savetxt(buffer, np.hstack([origins, directions]), fmt="%.10g")
    return buffer.getvalue().encode("ascii")


def _parse(out: bytes, expected: int):
    lines = [line for line in out.decode("ascii", errors="replace").splitlines() if line.strip()]
    if len(lines) != expected:
        raise ViewTraceError(f"rtrace returned {len(lines)} ray results, expected {expected}")
    lengths = np.empty(expected)
    points = np.full((expected, 3), np.nan)
    normals = np.full((expected, 3), np.nan)
    surfaces: list[str] = [""] * expected
    modifiers: list[str] = [""] * expected
    for i, line in enumerate(lines):
        tokens = line.split()
        try:
            length = float(tokens[0])
        except (ValueError, IndexError):
            raise ViewTraceError(f"rtrace result {i + 1}: cannot read a length from {line!r}") from None
        if not np.isfinite(length) or length >= MISS_LENGTH_MIN:
            lengths[i] = np.inf
            continue
        if len(tokens) < 9:
            raise ViewTraceError(
                f"rtrace result {i + 1}: expected length, point, normal, surface and modifier, got {line!r}"
            )
        lengths[i] = length
        try:
            points[i] = [float(t) for t in tokens[1:4]]
            normals[i] = [float(t) for t in tokens[4:7]]
        except ValueError:
            raise ViewTraceError(f"rtrace result {i + 1}: cannot read a point and a normal from {line!r}") from None
        surfaces[i] = tokens[7]
        modifiers[i] = tokens[8]
    return lengths, points, normals, surfaces, modifiers


def _run_rtrace(rad, octree, origins, directions, chunk_rays, processes):
    lengths, points, normals, surfaces, modifiers = [], [], [], [], []
    for start in range(0, origins.shape[0], chunk_rays):
        o = origins[start:start + chunk_rays]
        d = directions[start:start + chunk_rays]
        out = rad.run(["rtrace", "-n", str(process_count(processes)), *_RTRACE_ARGS, str(octree)], stdin=_rays_stdin(o, d))
        length, _printed_point, normal, surface, modifier = _parse(out, o.shape[0])
        # The hit point is computed from the segment's own start and the
        # printed length, not read from rtrace's printed x y z: rtrace
        # prints reals with about 7 significant digits (-faa), which can
        # round a hit several cm in survey coordinates -- far more than
        # NUDGE_M. The length is relative to the ray's own start, so its
        # rounding stays negligible. `_printed_point` is still parsed
        # above, to validate the line has one.
        computed = np.full_like(o, np.nan)
        finite = np.isfinite(length)
        computed[finite] = o[finite] + length[finite, None] * d[finite]
        lengths.append(length)
        points.append(computed)
        normals.append(normal)
        surfaces.extend(surface)
        modifiers.extend(modifier)
    return np.concatenate(lengths), np.concatenate(points), np.concatenate(normals), surfaces, modifiers


def _intern(name: str, names: list[str], index: dict[str, int]) -> int:
    if name not in index:
        index[name] = len(names)
        names.append(name)
    return index[name]


def _as_rays(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ViewTraceError(f"{label} must have shape (n, 3), got {array.shape}")
    return array


def _same_layer(gap: np.ndarray, prev_normal: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Two glass hits ``gap`` apart are one layer: within LAYER_MERGE_M
    along the ray, or on parallel surfaces within LAYER_MERGE_M measured
    perpendicular to them. The engine's copy of metrics.views.same_layer."""
    with np.errstate(invalid="ignore", divide="ignore"):
        close = np.linalg.norm(gap, axis=1) < LAYER_MERGE_M
        n0 = prev_normal / np.linalg.norm(prev_normal, axis=1, keepdims=True)
        n1 = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        parallel = np.abs(np.sum(n0 * n1, axis=1)) >= LAYER_PARALLEL_COS
        perpendicular = np.abs(np.sum(gap * n1, axis=1))
        return close | (parallel & (perpendicular < LAYER_MERGE_M))


def trace_views(
    rad: Radiance, octree, origins, directions, *,
    pass_through: frozenset[str], exterior: frozenset[str],
    skip_within_m: float, max_segments: int, chunk_rays: int = CHUNK_RAYS, processes: int = 1,
) -> RaySegments:
    """March every ray (module docstring). ``processes`` is rtrace's ``-n``:
    that many rtrace processes share each call's rays."""
    origins = _as_rays(origins, "origins")
    directions = _as_rays(directions, "directions")
    if origins.shape != directions.shape:
        raise ViewTraceError(f"origins {origins.shape} and directions {directions.shape} differ")
    if max_segments < 1:
        raise ViewTraceError(f"max_segments must be >= 1, got {max_segments!r}")
    if chunk_rays < 1:
        raise ViewTraceError(f"chunk_rays must be >= 1, got {chunk_rays!r}")
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise ViewTraceError(f"processes must be an integer >= 1, got {processes!r}")
    if not exterior <= pass_through:
        missing = sorted(exterior - pass_through)
        raise ViewTraceError(f"every exterior modifier must also be in pass_through; missing {missing}")

    n = origins.shape[0]
    modifier = np.full((n, max_segments), UNUSED, dtype=np.int32)
    surface = np.full((n, max_segments), -1, dtype=np.int32)
    hit = np.full((n, max_segments, 3), np.nan)
    normal = np.full((n, max_segments, 3), np.nan)
    modifier_names: list[str] = []
    modifier_index: dict[str, int] = {}
    surface_names: list[str] = []
    surface_index: dict[str, int] = {}

    current = origins.copy()
    exit_point = np.full((n, 3), np.nan)
    exit_normal = np.full((n, 3), np.nan)
    exited = np.zeros(n, dtype=bool)
    active = np.arange(n)

    for s in range(max_segments):
        if active.size == 0:
            break
        lengths, points, normals, surfs, mods = _run_rtrace(
            rad, octree, current[active], directions[active], chunk_rays, processes,
        )
        miss = ~np.isfinite(lengths)
        mi = np.array(
            [MISS if miss[j] else _intern(mods[j], modifier_names, modifier_index) for j in range(active.size)],
            dtype=np.int32,
        )
        si = np.array(
            [-1 if miss[j] else _intern(surfs[j], surface_names, surface_index) for j in range(active.size)],
            dtype=np.int32,
        )
        modifier[active, s] = mi
        surface[active, s] = si
        hit[active, s] = points
        normals[miss] = np.nan
        normal[active, s] = normals

        known = mi >= 0
        is_pass = np.array([name in pass_through for name in modifier_names], dtype=bool)
        is_exterior = np.array([name in exterior for name in modifier_names], dtype=bool)
        hit_pass = np.zeros(active.size, dtype=bool)
        hit_exterior = np.zeros(active.size, dtype=bool)
        hit_pass[known] = is_pass[mi[known]]
        hit_exterior[known] = is_exterior[mi[known]]

        was_exited = exited[active]
        with np.errstate(invalid="ignore"):
            from_exit = np.linalg.norm(points - exit_point[active], axis=1)
            before = ~was_exited & known & hit_pass
            after = was_exited & known & (
                (from_exit < skip_within_m)
                | (hit_pass & ~hit_exterior)
                | (hit_exterior & _same_layer(points - exit_point[active], exit_normal[active], normals))
            )
        newly_exited = ~was_exited & hit_exterior
        exit_point[active[newly_exited]] = points[newly_exited]
        exit_normal[active[newly_exited]] = normals[newly_exited]
        exited[active[newly_exited]] = True

        go_on = before | after
        continuing = active[go_on]
        current[continuing] = points[go_on] + NUDGE_M * directions[continuing]
        active = continuing

    return RaySegments(
        modifiers=tuple(modifier_names),
        surfaces=tuple(surface_names),
        modifier=modifier,
        surface=surface,
        hit=hit,
        normal=normal,
    )


class RadianceViewTracer:
    """The real tracer the views driver uses; tests inject a fake instead."""

    def __init__(self, rad: Radiance, processes: int = 1) -> None:
        self._rad = rad
        self._processes = processes

    @property
    def radiance_version(self) -> str:
        return self._rad.version

    def octree(self, scene, materials, work_dir) -> pathlib.Path:
        return build_view_octree(self._rad, scene, materials, work_dir)

    def trace(self, octree, origins, directions, *, pass_through, exterior, skip_within_m, max_segments) -> RaySegments:
        return trace_views(
            self._rad, octree, origins, directions,
            pass_through=pass_through, exterior=exterior,
            skip_within_m=skip_within_m, max_segments=max_segments, processes=self._processes,
        )
