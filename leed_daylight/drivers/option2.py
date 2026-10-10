# leed_daylight/drivers/option2.py
"""The ``option2`` command: LEED v4.1 BD+C EQ Credit Daylight, Option 2.

    python -m leed_daylight.drivers.option2 "<project folder>" [--case <name>] [--processes N]

It prepares the same model Option 1 uses (``<folder>/_run/``), then writes
``<folder>/_run/option2/<case>/option2_results.json`` and ``points.csv``,
and prints a summary. Working files go to ``<folder>/_run/option2/<case>/``
(``skies/``, ``engine/``) and ``<folder>/_run/option2/radiance.log``. A
stopped run resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..build.prepare import prepare
from ..constants import TOOL_VERSION
from ..engine.jobs import folder_lock
from ..engine.radiance import Radiance, RadianceError, find_binaries, subprocess_runner
from ..metrics.clearest_day import DayCandidate
from ..project.checks import check_option2
from ..project.config import load_project
from ..project.inputs import input_fingerprints
from ..project.layout import scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.rules import load_rules
from ..project.spaces import load_spaces
from ..project.weather import read_wea
from ..scene.reader import read_scene
from ..scene.validate import validate_scene
from ._folder import PREPARED_DIRS, REPO_ROOT, check_folder, check_output_dirs
from .run import _positive_int, _progress_printer, _raypath_dirs, _resolve_case
from .run_option2 import Option2Engine, Option2Result, plan_option2, run_option2

#: The Radiance binaries Option 2 runs.
OPTION2_BINARIES = ("oconv", "rtrace", "gendaylit", "xform")
_REQUIRED_FILES = ("project.yaml", "materials.csv", "spaces.csv")


class Option2CommandError(ValueError):
    """The option2 command cannot proceed from these inputs."""


def _check_folder(folder: pathlib.Path) -> None:
    check_folder(folder, REPO_ROOT, Option2CommandError)
    for name in _REQUIRED_FILES:
        if not (folder / name).is_file():
            raise Option2CommandError(f"{folder}: missing required file {name!r}")


def _candidate_json(c: DayCandidate | None) -> dict | None:
    if c is None:
        return None
    return {
        "date": f"{c.month:02d}-{c.day:02d}",
        "offset_days": c.offset_days,
        "direct_normal_wh_m2": c.direct_normal_wh,
        "perez_epsilon": c.epsilon,
    }


def _results_json(result: Option2Result, radiance_version: str | None) -> dict:
    return {
        "tool_version": TOOL_VERSION,
        "radiance_version": radiance_version,
        "case": result.case,
        "settings": result.settings,
        "days": [
            {
                "equinox": f"{d.equinox[0]:02d}-{d.equinox[1]:02d}",
                "chosen": _candidate_json(d.chosen),
                "top": [_candidate_json(c) for c in d.top],
                "perez_best": _candidate_json(d.epsilon_best),
                "hourly_direct_normal_w_m2": list(d.hourly_direct_normal),
                "hourly_diffuse_horizontal_w_m2": list(d.hourly_diffuse_horizontal),
            }
            for d in result.days
        ],
        "skies": [
            {
                "label": m.label, "month": m.month, "day": m.day, "hour": m.hour,
                "direct_normal_w_m2": m.dni, "diffuse_horizontal_w_m2": m.dhi,
                "sun_altitude_deg": m.altitude_deg, "sun_azimuth_deg": m.azimuth_deg,
            }
            for m in result.moments
        ],
        "spaces": [
            {
                "room": s.room, "area_m2": s.area_m2, "points": s.points,
                "passing": s.passing, "pass_pct": s.pass_pct, "n_low": s.n_low,
                "n_high": s.n_high, "upper_limit_applied": s.upper_limit_applied,
            }
            for s in result.spaces
        ],
        "area_pct": result.area_pct,
        "point_pct": result.point_pct,
        "score": {
            "points": result.score.points,
            "band": result.score.band,
            "reason": result.score.reason,
        },
        "warnings": list(result.warnings),
    }


def _write_points(path: pathlib.Path, result: Option2Result) -> None:
    labels = [m.label for m in result.moments]
    times = result.settings["times"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "room", "x", "y", "z", *(f"lux_{label}" for label in labels),
            *(f"average_{t}" for t in times), "passes",
        ])
        for row in result.point_rows:
            writer.writerow([
                row["room"], f"{row['x']:.3f}", f"{row['y']:.3f}", f"{row['z']:.3f}",
                *(f"{row['lux'][label]:.1f}" for label in labels),
                *(f"{row['average'][t]:.1f}" for t in times),
                "yes" if row["passes"] else "no",
            ])


def _print_summary(result: Option2Result, results_path: pathlib.Path, *, out) -> None:
    print(f"case: {result.case}", file=out)
    days = ", ".join(
        f"{d.chosen.month:02d}-{d.chosen.day:02d} (DNI {d.chosen.direct_normal_wh:,.0f} Wh/m2)"
        for d in result.days
    )
    print(f"clearest days: {days}", file=out)
    print("time basis: weather file local time, no daylight saving", file=out)
    print(
        f"passing floor area: {result.area_pct:.1f}% (points: {result.point_pct:.1f}%)",
        file=out,
    )
    if result.score.points is None:
        print(f"score: not awarded: {result.score.reason}", file=out)
    else:
        band = result.score.band if result.score.band is not None else "below the lowest band"
        unit = "point" if result.score.points == 1 else "points"
        print(f"score: {result.score.points} {unit} ({band})", file=out)
    for s in result.spaces:
        limit = "" if s.upper_limit_applied else "; no upper limit (view-preserving shades)"
        print(
            f"  {s.room}: {s.area_m2:.1f} m2, {s.pass_pct:.1f}% pass "
            f"({s.n_low} low, {s.n_high} high){limit}",
            file=out,
        )
    print(f"warnings: {len(result.warnings)}", file=out)
    for warning in result.warnings:
        print(f"  - {warning}", file=out)
    print(f"results written to {results_path}", file=out)


def option2_folder(
    folder, case: str | None = None, *, engine: Option2Engine | None = None,
    rad: Radiance | None = None, out=sys.stdout, processes: int = 1, progress: bool = False,
) -> pathlib.Path:
    """Run Option 2 on ``folder``; return the written ``option2_results.json``.
    With ``progress`` (the verb's own ``main``), one ``progress: <text>``
    line is printed to ``out`` after each sky (wizard spec 8.4)."""
    folder = pathlib.Path(folder)
    _check_folder(folder)
    project = load_project(folder / "project.yaml")
    # Spec 7: refused by name before any Radiance command runs, before the
    # prepared model is rewritten, and whether or not Radiance is installed.
    errors = check_option2(project)
    if errors:
        raise Option2CommandError("Option 2 cannot run: " + "; ".join(errors))
    weather = read_wea(project.weather_file)
    if abs(weather.time_zone - project.site.meridian_radiance) > 1e-6:
        raise Option2CommandError(
            f"the weather file's time_zone ({weather.time_zone:g}) differs from "
            f"project.yaml site.meridian_radiance ({project.site.meridian_radiance:g}); "
            "Option 2 checks the weather file's own hours, so both must name the same time zone"
        )
    # The clearest days and the sky at every checked time: no direct sun near
    # an equinox, a sun below the horizon and zero diffuse light refuse here.
    plan_option2(project, weather)
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(folder / "materials.csv")
    validate_scene(
        prims, project.scene.units, roles, declared_types=declared_types(folder / "materials.csv"),
    ).raise_if_errors()
    rooms = scene_rooms(project, prims, roles)
    spaces = load_spaces(folder / "spaces.csv")
    case = _resolve_case(spaces, case)
    rules = load_rules(project.credit_rules_path)
    # Final review I2: fingerprint the inputs as they were read, before the
    # simulation, so an edit made during the run leaves the results stale.
    inputs = input_fingerprints(project, folder)
    # A missing Radiance is reported before _run is rewritten.
    needs_radiance = rad is None and engine is None
    if needs_radiance:
        find_binaries(names=OPTION2_BINARIES)
    # Re-review OOS-1: the model writer and the per-case sky and trace
    # caches delete files under these folders; a link at one is refused
    # before anything changes.
    check_output_dirs(
        folder, *(("_run", *parts) for parts in PREPARED_DIRS),
        ("_run", "option2", case, "skies"), ("_run", "option2", case, "engine"),
    )
    prepared = prepare(project, prims, roles, rooms, spaces, case, folder / "_run")

    work_dir = folder / "_run" / "option2"
    if needs_radiance:
        work_dir.mkdir(parents=True, exist_ok=True)
        rad = Radiance(
            subprocess_runner(search_dirs=_raypath_dirs(project, prepared.root)),
            work_dir / "radiance.log",
        )

    # Skies and cached traces live per case, so switching cases keeps both caches.
    case_dir = work_dir / case
    result = run_option2(
        rad, project, prepared, spaces, rules, case, weather=weather, engine=engine,
        work_dir=case_dir, processes=processes,
        progress=_progress_printer(out) if progress else None,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    results_path = case_dir / "option2_results.json"
    payload = _results_json(result, rad.version if rad is not None else None)
    payload["inputs"] = inputs
    results_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    _write_points(case_dir / "points.csv", result)
    _print_summary(result, results_path, out=out)
    return results_path


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.option2",
        description=(
            "Run LEED v4.1 BD+C EQ Credit Daylight, Option 2 (illuminance at 9:00 and "
            "15:00 on the clearest day near each equinox), and write "
            "option2_results.json and points.csv."
        ),
    )
    parser.add_argument("folder", help="the project folder (project.yaml, materials.csv, spaces.csv)")
    parser.add_argument("--case", default=None, help="the case to run (default: the first case in spaces.csv)")
    parser.add_argument(
        "--processes", type=_positive_int, default=1, metavar="N",
        help="run rtrace with N processes (-n N); same results, faster (default: 1)",
    )
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, Option2CommandError)
        with folder_lock(pathlib.Path(args.folder), "option2"):
            option2_folder(args.folder, args.case, processes=args.processes, progress=True)
    except (ValueError, RadianceError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
