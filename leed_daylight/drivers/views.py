# leed_daylight/drivers/views.py
"""The `views` verb (layer 7): LEED v4.1 BD+C EQ Credit Quality Views on
one project folder::

    .venv/bin/python -m leed_daylight.drivers.views "<project folder>" [--case <name>] [--processes N]

Chain: load project.yaml (it must have a `views:` section), the scene,
materials.csv and spaces.csv -> prepare the model, with the views ticks as
its case, into _run/views/prepared/ (every exterior pane gets its frame
whatever the case) -> the views model (see-through classes, black
blind-free scene, eye-height grids, pane tops) -> one octree -> per room,
the panorama rays of every point, marched by the tracer in batches of
POINT_BATCH points -> Type 2, Type 3 and Quality per point, space and
project -> views_results.json, points.csv and SVG plans in
_run/views/<case>/.

Each room's per-point result is cached in _run/views/engine/ under a
fingerprint of the octree, the points, the rays and every setting, so a
stopped run resumes and a changed setting recomputes.

``tracer`` is the test seam, as ``engine`` is for ``run``: when it is
given, no Radiance binary is looked for or run.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import yaml
from shapely.errors import GEOSException

from ..build.prepare import prepare
from ..build.view_model import (
    blocked_clear_warnings, blocked_panes, copy_rows, ground_plane, pane_tops, room_file_stem,
    view_classes, view_grids, vision_panes, write_view_scene,
)
from ..build.view_rays import panorama, rays_for_points
from ..constants import TOOL_VERSION, UNVERIFIED
from ..deliver.view_plans import plan_svg
from ..engine.fingerprint import delete_sidecar, fingerprint_of, is_current, write_sidecar
from ..engine.jobs import PROGRESS_PREFIX, folder_lock, replace_from_part
from ..engine.radiance import Radiance, RadianceError, find_binaries, subprocess_runner
from ..engine.view_trace import (
    LAYER_MERGE_M, LAYER_PARALLEL_COS, MISS_LENGTH_MIN, NUDGE_M, RadianceViewTracer, RaySegments,
)
from ..metrics import views as mv
from ..project.config import load_project
from ..project.inputs import input_fingerprints
from ..project.layout import SpaceLayout, lay_out, scene_rooms
from ..project.materials import EXTERIOR_GLAZING_ROLES, MaterialRow, declared_types, load_materials, vlt_warnings
from ..project.materials import rows as material_rows
from ..project.rules import load_rules, view_rules, views_verified
from ..project.spaces import (
    VIEWS_PREPARE_CASE, load_spaces, match_to_rooms, space_rows, views_case_names, views_included,
    views_prepare_spaces,
)
from ..scene.primitives import Primitive
from ..scene.reader import read_scene
from ..scene.validate import validate_scene
from ._folder import PREPARED_DIRS, REPO_ROOT, check_folder, check_output_dirs

#: Segments per ray: interior glass, exterior glass, near objects skipped,
#: the object seen. Eight leaves room for double faces on each.
MAX_SEGMENTS = 8
#: Points traced per tracer call (80 x 5,580 rays at the 2 deg default).
POINT_BATCH = 80
#: Bump this whenever metrics.views or engine.view_trace changes a result,
#: so the per-room cache (fingerprinted below) is never reused stale.
VIEWS_METHOD_VERSION = 2

_REQUIRED_FILES = ("project.yaml", "materials.csv", "spaces.csv")
_LAYERS = ("type2", "type3", "quality")
_ATRIUM_WARNING = (
    "the 30 % cap on views into interior atria is not assessed; a view through "
    "interior glass counts like any other"
)
#: Roof-glass design 5.11 (D7): only perimeter clear glass is vision glazing.
ROOF_GLASS_NOTE = (
    "Roof glass (skylights, roof monitors, sloped roof glass) and diffusing glass are not "
    "vision glazing: they block view rays and set no head height."
)
_METHOD_NOTES = (
    "Head height: top of the vision glass above the finished floor of the point's storey.",
    "Eye height: views.eye_height_m, with the source given in views.eye_height_source.",
    "Objects at 7.5 m: solid-angle-weighted median outside distance over a 360 x 60 degree "
    "panorama (ClimateStudio method, not LEED text).",
    "Objects closer than views.method.min_outside_distance_m beyond the glass are looked "
    "through (ClimateStudio default, not LEED text).",
    "Interior glass: a view may cross at most views.max_interior_glass partitions.",
    ROOF_GLASS_NOTE,
)


class ViewsError(ValueError):
    """The views command cannot proceed from these inputs."""


class ViewTracer(Protocol):
    radiance_version: str | None

    def octree(self, scene: pathlib.Path, materials: pathlib.Path, work_dir: pathlib.Path) -> pathlib.Path: ...

    def trace(self, octree, origins, directions, *, pass_through, exterior, skip_within_m, max_segments) -> RaySegments: ...


def _check_folder(folder: pathlib.Path) -> None:
    check_folder(folder, REPO_ROOT, ViewsError)
    for name in _REQUIRED_FILES:
        if not (folder / name).is_file():
            raise ViewsError(f"{folder}: missing required file {name!r}")


def _glazing_family(role: str | None) -> str:
    if role in EXTERIOR_GLAZING_ROLES:
        return "exterior"
    if role == "glazing_interior":
        return "interior"
    return "none"


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _room_points(tracer, octree, grid, directions, weights, *, classes, roles, tops, pass_through,
                 exterior, skip, z_ground, max_interior_glass, head_multiple, object_distance,
                 cache_dir: pathlib.Path) -> mv.PointViews:
    cache = cache_dir / f"{room_file_stem(grid.room)}.npz"
    fp = fingerprint_of({"octree": octree}, {
        "views_method": VIEWS_METHOD_VERSION,
        "points_sha256": _sha(grid.points),
        "directions_sha256": _sha(directions),
        "weights_sha256": _sha(weights),
        "pass_through": sorted(pass_through),
        "exterior": sorted(exterior),
        "classes": {k: classes[k] for k in sorted(classes)},
        "glazing_family": {m: _glazing_family(roles.get(m)) for m in sorted(pass_through)},
        "pane_tops": {k: float(tops[k]) for k in sorted(tops)},
        "level_z": float(grid.level_z),
        "ground_z": float(z_ground),
        "max_interior_glass": int(max_interior_glass),
        "skip_within_m": float(skip),
        "head_height_multiple": float(head_multiple),
        "object_distance_m": float(object_distance),
        "max_segments": MAX_SEGMENTS,
        "miss_length_min": MISS_LENGTH_MIN,
        "nudge_m": NUDGE_M,
        "layer_merge_m": LAYER_MERGE_M,
        "layer_parallel_cos": LAYER_PARALLEL_COS,
        "radiance_version": tracer.radiance_version,
    })
    if is_current(cache, fp):
        with np.load(cache) as data:
            return mv.PointViews(
                type2=data["type2"], type3=data["type3"], median_outside_m=data["median_outside_m"],
                reaching_rays=data["reaching_rays"], limit_rays=data["limit_rays"],
            )
    delete_sidecar(cache)
    parts = []
    for start in range(0, grid.points.shape[0], POINT_BATCH):
        origins, dirs = rays_for_points(grid.points[start:start + POINT_BATCH], directions)
        segments = tracer.trace(
            octree, origins, dirs, pass_through=pass_through, exterior=exterior,
            skip_within_m=skip, max_segments=MAX_SEGMENTS,
        )
        kind_by_modifier = [
            mv.kind_of(classes.get(name, "blocks"), _glazing_family(roles.get(name)))
            for name in segments.modifiers
        ]
        kinds = mv.segment_kinds(segments.modifier, kind_by_modifier)
        pane_top = np.array([tops.get(name, np.nan) for name in segments.surfaces], dtype=float)
        outcomes = mv.ray_outcomes(
            origins, dirs, kinds, segments.surface, segments.hit, pane_top,
            level_z=grid.level_z, max_interior_glass=max_interior_glass,
            skip_within_m=skip, ground_z=z_ground, normals=segments.normal,
        )
        parts.append(mv.point_views(
            outcomes, weights, head_height_multiple=head_multiple, object_distance_m=object_distance,
        ))
    result = mv.PointViews.concatenate(parts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, type2=result.type2, type3=result.type3, median_outside_m=result.median_outside_m,
             reaching_rays=result.reaching_rays, limit_rays=result.limit_rays)
    write_sidecar(cache, fp)
    return result


def _median_text(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def _write_points(path: pathlib.Path, grids, results) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "storey", "x", "y", "z", "type2", "type3", "quality",
                         "median_outside_m", "reaching_rays", "limit_rays"])
        for grid, points in zip(grids, results):
            quality = points.quality
            for i, (x, y, z) in enumerate(grid.points):
                writer.writerow([
                    grid.room, grid.storey, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                    int(points.type2[i]), int(points.type3[i]), int(quality[i]),
                    _median_text(float(points.median_outside_m[i])),
                    int(points.reaching_rays[i]), int(points.limit_rays[i]),
                ])


#: The only Radiance binaries a views run executes.
VIEWS_BINARIES: tuple[str, ...] = ("oconv", "rtrace")


def _raypath_dirs(project, model_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """RAYPATH search folders, as in the run command: each scene file's own
    folder (in ``scene.files`` order, each once), then the views model
    folder, so a relative ``.cal``/``.xml``/data reference resolves."""
    dirs: list[pathlib.Path] = []
    for path in (*(pathlib.Path(f).parent for f in project.scene.files), model_dir):
        resolved = pathlib.Path(path).resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    return tuple(dirs)


def _write_plans(case_dir: pathlib.Path, grids, results, spaces_views, layout: SpaceLayout,
                 north_rotation_deg: float) -> None:
    """One SVG plan per storey and layer; every piece of each space is
    outlined (merge-split spec 7), with no label and no split line."""
    for storey in sorted({g.storey for g in grids}):
        index = [i for i, g in enumerate(grids) if g.storey == storey]
        pieces = [piece for i in index for piece in layout.space(grids[i].room).pieces]
        outlines = [ring for r in pieces for ring in (r.outline, *r.holes)]
        points_xy = np.vstack([grids[i].points[:, :2] for i in index])
        area = sum(spaces_views[i].area_m2 for i in index)
        for layer in _LAYERS:
            flags = np.concatenate([getattr(results[i], layer) for i in index])
            passing = sum(spaces_views[i].area_m2 * getattr(spaces_views[i], f"{layer}_pct") / 100.0 for i in index)
            title = f"Storey {storey}, {layer}: {100.0 * passing / area:.1f} % of floor area"
            (case_dir / f"plan_{storey}_{layer}.svg").write_text(plan_svg(
                title=title, outlines=outlines, points_xy=points_xy, passed=flags,
                cell_m=grids[index[0]].spacing_m, north_rotation_deg=north_rotation_deg,
            ), encoding="utf-8", newline="\n")


def with_scene_types(rows: Sequence[MaterialRow], prims: Sequence[Primitive]) -> list[MaterialRow]:
    """``rows`` with each blank ``primitive_type`` filled from the scene's
    material of that name (left blank when the scene defines none). The
    materials.csv cell may be left empty; the view classes must still know
    a ``trans`` (frosted glass) from a ``glass``."""
    scene_types = {p.name: p.ptype for p in prims if not p.is_surface}
    return [
        row if row.primitive_type
        else dataclasses.replace(row, primitive_type=scene_types.get(row.modifier, ""))
        for row in rows
    ]


def run_views(folder, case: str | None = None, *, tracer: ViewTracer | None = None, out=sys.stdout,
              processes: int = 1, progress: bool = False) -> pathlib.Path:
    """Run Quality Views on ``folder``; return the path of views_results.json.

    ``processes`` is how many rtrace processes trace the rays (rtrace -n);
    it changes the speed, never a result. With ``progress`` (the verb's own
    ``main``), one ``progress: space <i> of <n>`` line is printed to ``out``
    after each space is traced (wizard spec 8.4).
    """
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise ViewsError(f"--processes must be an integer >= 1, got {processes!r}")
    folder = pathlib.Path(folder)
    _check_folder(folder)

    project = load_project(folder / "project.yaml")
    if project.views is None:
        raise ViewsError(
            "project.yaml has no views: section; add views.eye_height_m and "
            "views.eye_height_source to run Quality Views"
        )
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    # Final review I2: a blank primitive_type cell takes the scene's own
    # type, so frosted interior glass blocks views whatever the cell says.
    rows = with_scene_types(material_rows(folder / "materials.csv"), prims)
    validate_scene(
        prims, project.scene.units, roles, declared_types=declared_types(folder / "materials.csv"),
    ).raise_if_errors()
    rooms = scene_rooms(project, prims, roles)
    spaces = load_spaces(folder / "spaces.csv")
    known = views_case_names(spaces)
    if not known:
        views_included(spaces, "")  # the named refusal: no views_<case> column
    case = known[0] if case is None else case
    ticked = views_included(spaces, case)
    if not ticked:
        raise ViewsError(f"views case {case!r} ticks no space")

    rules = view_rules(load_rules(project.credit_rules_path))
    # Final review I2: fingerprint the inputs as they were read, before the
    # ray tracing, so an edit made during the run leaves the results stale.
    inputs = input_fingerprints(project, folder)
    for key in ("vlt_min_exclusive", "head_height_multiple", "object_distance_m"):
        if rules[key] == UNVERIFIED:
            raise ViewsError(f"credit_rules.yaml views.{key} is {UNVERIFIED}; it is needed to compute Quality Views")
    method = rules["method"]
    skip = float(method["min_outside_distance_m"])
    directions, weights = panorama(project.views.angular_step_deg, float(method["panorama_vertical_deg"]))

    run_dir = folder / "_run" / "views"
    # Re-review OOS-1: the model writer and the room caches delete files
    # under these folders; a link at one is refused before anything changes.
    check_output_dirs(
        folder, *(("_run", "views", "prepared", *parts) for parts in PREPARED_DIRS), ("_run", "views", "engine"),
    )
    case_dir = run_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    # Roof-glass design 4.3: view rays are traced with -ab 0 by design, so
    # the diffusing-glass -ab check does not apply here.
    prepared = prepare(project, prims, roles, rooms, views_prepare_spaces(spaces, case), VIEWS_PREPARE_CASE,
                       run_dir / "prepared", check_ambient=False)
    # Material copies (<modifier>__sloped, __skylight) carry their
    # original's role (effective_roles) and view tag (copy_rows).
    view_roles = {**roles, **prepared.effective_roles}
    vt = {m: record["visible_transmittance"] for m, record in prepared.manifest["glazing"].items()}
    classes = view_classes(
        [*rows, *copy_rows(rows, prepared.effective_roles)], vt,
        vlt_min_exclusive=float(rules["vlt_min_exclusive"]),
    )
    pass_through = frozenset(m for m, c in classes.items() if c == "clear")
    exterior = frozenset(m for m in pass_through if view_roles.get(m) in EXTERIOR_GLAZING_ROLES)
    if not exterior:
        raise ViewsError(
            "no exterior glazing is see-through for views (visible transmittance above "
            f"{rules['vlt_min_exclusive']}, or marked clear); no point can have a view"
        )
    vision = vision_panes(prepared)
    scene_file, materials_file = write_view_scene(
        prepared, view_roles, moveable_furniture=project.analysis.moveable_furniture, out_dir=run_dir / "model",
        blocked=blocked_panes(prepared),
    )
    matched = match_to_rooms(spaces, rooms.rooms)
    layout = lay_out(spaces, matched)
    grids = view_grids(spaces, matched, project.grid, project.views.eye_height_m, case, layout=layout)
    tops = pane_tops(prepared)
    z_ground, ground_warnings = ground_plane(
        prims, roles, storey_levels=[r.level_z for r in rooms.rooms],
        room_levels={g.room: g.level_z for g in grids},
    )

    if tracer is None:
        find_binaries(names=VIEWS_BINARIES)
        runner = subprocess_runner(search_dirs=_raypath_dirs(project, run_dir / "model"))
        tracer = RadianceViewTracer(Radiance(runner, run_dir / "radiance.log"), processes=processes)
    octree = tracer.octree(scene_file, materials_file, run_dir / "engine")

    results = []
    for number, grid in enumerate(grids, start=1):
        results.append(_room_points(
            tracer, octree, grid, directions, weights, classes=classes, roles=view_roles, tops=tops,
            pass_through=pass_through, exterior=exterior, skip=skip, z_ground=z_ground,
            max_interior_glass=project.views.max_interior_glass,
            head_multiple=float(rules["head_height_multiple"]),
            object_distance=float(rules["object_distance_m"]),
            cache_dir=run_dir / "engine",
        ))
        if progress:
            print(f"{PROGRESS_PREFIX}space {number} of {len(grids)}", file=out, flush=True)
    spaces_views = [mv.space_views(g.room, g.room_area_m2, r) for g, r in zip(grids, results)]
    threshold = rules["threshold_pct"] if views_verified(rules) else UNVERIFIED
    project_result = mv.project_views(spaces_views, threshold_pct=threshold, points=int(rules["points"]))

    warnings = list(project.warnings) + list(rooms.warnings) + [_ATRIUM_WARNING] + ground_warnings
    warnings.extend(vlt_warnings(folder / "materials.csv"))
    ticked_names = {s.room for s in ticked}
    # Space rows only: a piece row (merge_into) is part of its space.
    excluded = [s for s in space_rows(spaces) if s.regularly_occupied and s.room not in ticked_names]
    if excluded:
        listed = ", ".join(f"{s.room!r} ({s.views_reason or 'no reason given'})" for s in excluded)
        warnings.append(
            f"{len(excluded)} regularly occupied space(s) not ticked in views_{case}, so left out "
            f"of the Quality Views floor area: {listed}; LEED v4.1 allows excluding only certain "
            "space types"
        )
    for row in rows:
        if row.view == "clear" and roles.get(row.modifier) not in EXTERIOR_GLAZING_ROLES | {"glazing_interior"}:
            warnings.append(
                f"{row.modifier}: marked 'clear' in materials.csv although its role is "
                f"{roles.get(row.modifier)!r}; views pass through it"
            )
    warnings.extend(blocked_clear_warnings(prepared, rows))
    storeys_with_glass = {
        g.storey for g in prepared.groups
        if any(o.name in vision and o.modifier in exterior for o in g.openings)
    }
    for storey in sorted({g.storey for g in grids} - storeys_with_glass):
        warnings.append(f"storey {storey}: ticked spaces but no exterior vision glass on this storey")
    for grid, points in zip(grids, results):
        limit = int(points.limit_rays.sum())
        if limit:
            warnings.append(
                f"{grid.room}: {limit} ray(s) stopped at the {MAX_SEGMENTS}-segment limit; they "
                "count as not reaching glass, or as an outside distance of 0 m if they had "
                "already passed it"
            )

    payload = {
        "tool_version": TOOL_VERSION,
        "radiance_version": tracer.radiance_version,
        "case": case,
        "settings": {
            "eye_height_m": project.views.eye_height_m,
            "eye_height_source": project.views.eye_height_source,
            "max_interior_glass": project.views.max_interior_glass,
            "angular_step_deg": project.views.angular_step_deg,
            "rays_per_point": int(directions.shape[0]),
            "ground_z_m": z_ground,
            "moveable_furniture": project.analysis.moveable_furniture,
            "rules": rules,
        },
        "method_notes": list(_METHOD_NOTES),
        "quality_area_pct": project_result.quality_area_pct,
        "quality_point_pct": project_result.quality_point_pct,
        "type2_area_pct": project_result.type2_area_pct,
        "type3_area_pct": project_result.type3_area_pct,
        "total_area_m2": project_result.total_area_m2,
        "quality_area_m2": project_result.quality_area_m2,
        "score": {
            "points": project_result.score.points,
            "band": project_result.score.band,
            "reason": project_result.score.reason,
        },
        "spaces": [
            {
                "room": s.room, "storey": g.storey, "area_m2": s.area_m2, "points": s.points,
                "type2_pct": s.type2_pct, "type3_pct": s.type3_pct,
                "quality_pct": s.quality_pct, "quality_area_m2": s.quality_area_m2,
            }
            for s, g in zip(spaces_views, grids)
        ],
        "excluded_spaces": [{"room": s.room, "reason": s.views_reason} for s in excluded],
        "warnings": warnings,
        "inputs": inputs,
    }
    results_path = case_dir / "views_results.json"
    # Both files are written as .part first and renamed only when both are whole, so a
    # failure while writing points.csv never leaves a new results file beside an old one.
    results_part = results_path.with_name(results_path.name + ".part")
    points_path = case_dir / "points.csv"
    points_part = points_path.with_name(points_path.name + ".part")
    try:
        results_part.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8", newline="\n")
        _write_points(points_part, grids, results)
        replace_from_part(results_part, results_path)
        try:
            replace_from_part(points_part, points_path)
        except BaseException:
            # The pair is all or nothing: no new results file beside an old points.csv.
            results_path.unlink(missing_ok=True)
            raise
    finally:
        results_part.unlink(missing_ok=True)
        points_part.unlink(missing_ok=True)
    _write_plans(case_dir, grids, results, spaces_views, layout, project.site.north_rotation_deg)

    score_text = (
        f"not awarded: {project_result.score.reason}" if project_result.score.points is None
        else f"{project_result.score.points} point(s): {project_result.score.reason}"
    )
    print(f"Quality Views, case {case}", file=out)
    print(
        f"quality view (floor area): {project_result.quality_area_pct:.1f}% "
        f"(points: {project_result.quality_point_pct:.1f}%)",
        file=out,
    )
    print(f"type 2: {project_result.type2_area_pct:.1f}%, type 3: {project_result.type3_area_pct:.1f}%", file=out)
    print(f"score: {score_text}", file=out)
    for s in spaces_views:
        print(f"  {s.room}: {s.area_m2:.1f} m2, quality {s.quality_pct:.1f}%", file=out)
    print(f"warnings: {len(warnings)}", file=out)
    for warning in warnings:
        print(f"  - {warning}", file=out)
    print(f"results written to {results_path}", file=out)
    return results_path


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.views",
        description="Run LEED v4.1 BD+C EQ Credit Quality Views on a project folder.",
    )
    parser.add_argument("folder", help="the project folder (project.yaml, materials.csv, spaces.csv)")
    parser.add_argument("--case", default=None, help="the views case (default: the first views_<case> column)")
    parser.add_argument("--processes", type=int, default=1,
                        help="rtrace processes tracing the rays (rtrace -n; default 1)")
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, ViewsError)
        with folder_lock(pathlib.Path(args.folder), "views"):
            run_views(args.folder, args.case, processes=args.processes, progress=True)
    except (ValueError, RadianceError, OSError, yaml.YAMLError, GEOSException) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
