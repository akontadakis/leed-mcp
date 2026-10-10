# leed_daylight/drivers/run.py
"""The `run` verb (layer 7, task RUN): one line, one Radiance model, one
results.json.

The full CLI (plan 3's ``cli.py``) is not built yet; this is the minimal
`run` verb only, so a person can put a Radiance model in a folder and get
the LEED v4.1 EQ Credit Daylight, Option 1 result on screen and on disk::

    .venv/bin/python -m leed_daylight.drivers.run "<project folder>" [--case <name>] [--processes N]

``--processes N`` (default 1) runs ``rfluxmtx``/``rcontrib`` with ``-n N``:
faster on a multi-core machine; same method, results equal within Monte
Carlo noise.

``run_folder`` runs exactly the chain ``tests/acceptance/conftest.py`` runs
(read it): ``load_project`` -> ``read_scene`` -> ``load_materials`` ->
``validate_scene(...).raise_if_errors()`` BEFORE ``scene_rooms`` ->
``load_spaces`` -> ``prepare`` into ``<folder>/_run/`` -> ``load_rules`` ->
``run_option1``. It writes
``<folder>/_run/option1/<case>/results.json (and points.csv beside it)``
and returns the results path; ``main(argv)`` is the thin CLI wrapper that
parses arguments, calls it, and turns any refusal into a one-line
``error: ...`` on stderr and exit code 1 -- never a traceback, and never a
broader catch than the chain's own refusal types.

``engine`` is a test-only seam (the same ``Option1Engine`` protocol
``run_option1`` defines): when it is given, this module never builds a real
``Radiance`` instance and never calls ``find_binaries()`` or
``rtrace -version`` -- there would be nothing for a real Radiance run to do,
and rule 10 (never execute a Radiance binary, not even ``-version``) applies
here exactly as it does to every other test in this codebase. This is also
why ``results.json``'s ``radiance_version`` is ``null`` whenever an engine
was injected and no ``rad`` was given directly: the value can only ever come
from an actual ``rad.version`` read, and that read is what an injected
engine exists to avoid.
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

from ..build.groups import WindowGroup
from ..build.prepare import prepare
from ..constants import TOOL_VERSION
from ..engine.aux_files import raypath_dirs as _raypath_dirs
from ..engine.jobs import PROGRESS_PREFIX, folder_lock
from ..engine.radiance import Radiance, RadianceError, find_binaries, subprocess_runner
from ..project.config import load_project
from ..project.inputs import input_fingerprints
from ..project.layout import scene_rooms
from ..project.materials import declared_types, load_materials
from ..project.rules import load_rules
from ..project.spaces import case_names, load_spaces
from ..scene.reader import read_scene
from ..scene.validate import validate_scene
from ._folder import PREPARED_DIRS, REPO_ROOT, check_folder, check_output_dirs
from .run_option1 import Option1Engine, Option1Result, run_option1

#: ``REPO_ROOT`` above is imported (not redefined) from the shared
#: ``_folder`` helper (spec 12: client models must never land in the
#: repository). A test that needs a folder "inside the repository" without
#: touching the real one monkeypatches this module's own ``REPO_ROOT``
#: rather than creating anything under the real repository root --
#: ``_check_folder`` below reads it as a module global at call time, so the
#: patched value is seen.

#: The three files a project folder must hold (checked, and named, in this
#: order).
_REQUIRED_FILES = ("project.yaml", "materials.csv", "spaces.csv")


class RunError(ValueError):
    """The run command cannot proceed from these inputs."""


def _check_folder(folder: pathlib.Path) -> None:
    check_folder(folder, REPO_ROOT, RunError)


def _check_required_files(folder: pathlib.Path) -> None:
    for name in _REQUIRED_FILES:
        if not (folder / name).is_file():
            raise RunError(f"{folder}: missing required file {name!r}")


def _resolve_case(spaces, case: str | None) -> str:
    known = case_names(spaces)
    if case is None:
        if not known:
            raise RunError("spaces.csv: no cases found")
        return known[0]
    if case not in known:
        raise RunError(f"unknown case {case!r}; known cases are {', '.join(known)}")
    return case


def _space_json(space) -> dict:
    return {
        "room": space.room,
        "area_m2": space.area_m2,
        "points": space.points,
        "sda_pct": space.sda_pct,
        "ase_pct": space.ase_pct,
        "ase_band": space.ase_band,
        "needs_glare_narrative": space.needs_glare_narrative,
        "residual_sky_pct": space.residual_sky_pct,
    }


def _group_json(group: WindowGroup) -> dict:
    """One window group as results.json lists it (roof-glass design 5.9).
    ``azimuth_bin_deg`` is None (JSON null) for skylight and diffusing
    groups; a diffusing group has no blind."""
    return {
        "group_id": group.group_id,
        "room": group.room,
        "kind": group.kind,
        "tilt_deg": group.tilt_deg,
        "azimuth_bin_deg": group.azimuth_bin_deg,
        "blind": group.kind != "diffusing",
    }


def _results_json(
    result: Option1Result, radiance_version: str | None, groups: Sequence[WindowGroup],
) -> dict:
    return {
        "tool_version": TOOL_VERSION,
        "radiance_version": radiance_version,
        "case": result.case,
        "occupied_hours": result.occupied_hours,
        "headline": result.headline,
        "sda_area_pct": result.sda_area_pct,
        "sda_point_pct": result.sda_point_pct,
        "ase_area_pct": result.ase_area_pct,
        "ase_point_pct": result.ase_point_pct,
        "score": {
            "points": result.score.points,
            "band": result.score.band,
            "additional_point_awarded": result.score.additional_point_awarded,
            "reason": result.score.reason,
        },
        "narrative_spaces": list(result.narrative_spaces),
        "spaces": [_space_json(s) for s in result.spaces],
        "blinds_closed_hours": {
            group_id: int((~schedule).sum())
            for group_id, schedule in result.blind_schedule.items()
        },
        "groups": [_group_json(g) for g in groups],
        "warnings": list(result.warnings),
        "settings": {
            "daylight_saving_year": result.daylight_saving_year,
        },
    }


def _write_points(path: pathlib.Path, result: Option1Result) -> None:
    """One row per sensor point (reports spec 3.1)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "storey", "x", "y", "z", "da_pct", "sda_pass", "ase_hours", "ase_exceeds"])
        for row in result.point_rows:
            writer.writerow([
                row["room"], row["storey"], f"{row['x']:.3f}", f"{row['y']:.3f}", f"{row['z']:.3f}",
                f"{row['da_pct']:.2f}", "yes" if row["sda_pass"] else "no",
                row["ase_hours"], "yes" if row["ase_exceeds"] else "no",
            ])


def _print_summary(result: Option1Result, results_path: pathlib.Path, *, out) -> None:
    other = "points" if result.headline == "floor_area" else "floor_area"
    if result.headline == "floor_area":
        sda_headline, sda_other = result.sda_area_pct, result.sda_point_pct
        ase_headline, ase_other = result.ase_area_pct, result.ase_point_pct
    else:
        sda_headline, sda_other = result.sda_point_pct, result.sda_area_pct
        ase_headline, ase_other = result.ase_point_pct, result.ase_area_pct

    print(f"case: {result.case}", file=out)
    print(f"occupied hours: {result.occupied_hours}", file=out)
    print(
        f"sDA ({result.headline}): {sda_headline:.1f}% "
        f"({other}: {sda_other:.1f}%)",
        file=out,
    )
    print(
        f"ASE ({result.headline}): {ase_headline:.1f}% "
        f"({other}: {ase_other:.1f}%)",
        file=out,
    )
    if result.score.points is None:
        print(f"score: not awarded: {result.score.reason}", file=out)
    else:
        awarded = " (additional point awarded)" if result.score.additional_point_awarded else ""
        # No band reached: say so, never print the missing band as "None".
        band = result.score.band if result.score.band is not None else "below the lowest band"
        print(f"score: {result.score.points} points ({band}){awarded}", file=out)
        print(f"  {result.score.reason}", file=out)

    for space in result.spaces:
        narrative = "; glare narrative needed" if space.needs_glare_narrative else ""
        print(
            f"  {space.room}: {space.area_m2:.1f} m2, sDA {space.sda_pct:.1f}%, "
            f"ASE {space.ase_pct:.1f}% ({space.ase_band}){narrative}",
            file=out,
        )

    print(f"warnings: {len(result.warnings)}", file=out)
    for warning in result.warnings:
        print(f"  - {warning}", file=out)

    print(f"results written to {results_path}", file=out)


def run_folder(
    folder,
    case: str | None = None,
    *,
    engine: Option1Engine | None = None,
    rad: Radiance | None = None,
    out=sys.stdout,
    processes: int = 1,
    progress: bool = False,
) -> pathlib.Path:
    """Run LEED Option 1 on ``folder`` and write
    ``<folder>/_run/option1/<case>/results.json (and points.csv beside it)``.
    With ``progress`` (the verb's own ``main``), one ``progress: <text>``
    line is printed to ``out`` after each unit of work (wizard spec 8.4).

    ``engine`` and ``rad`` are the same test seam ``run_option1`` defines.
    A real ``Radiance`` instance (via ``find_binaries()`` and
    ``subprocess_runner()``) is built only when BOTH are ``None`` -- the
    ordinary CLI path, never exercised by a test that injects a fake
    engine. That runner prepends the scene files' folder(s) and the
    prepared model folder to ``RAYPATH``. Returns the path to the written
    ``results.json``.
    """
    if not isinstance(folder, (str, pathlib.Path)):
        raise RunError(
            f"folder: expected a str or pathlib.Path, got "
            f"{type(folder).__name__}: {folder!r}"
        )
    folder = pathlib.Path(folder)
    _check_folder(folder)
    _check_required_files(folder)

    project = load_project(folder / "project.yaml")
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
    # (possibly hours-long) simulation, so an edit made during the run
    # leaves these results stale instead of looking current.
    inputs = input_fingerprints(project, folder)

    out_dir = folder / "_run"
    # Re-review OOS-1: the model writer and the engine caches delete files
    # under these folders; a link at one is refused before anything changes.
    check_output_dirs(
        folder, *(("_run", *parts) for parts in PREPARED_DIRS),
        ("_run", "engine", "sky"), ("_run", "engine", "groups"),
    )
    prepared = prepare(project, prims, roles, rooms, spaces, case, out_dir)

    if rad is None and engine is None:
        find_binaries()
        rad = Radiance(
            subprocess_runner(search_dirs=_raypath_dirs(project, prepared.root)),
            out_dir / "radiance.log",
        )

    result = run_option1(
        rad, project, prepared, spaces, rules, case, engine=engine, processes=processes,
        progress=_progress_printer(out) if progress else None,
    )
    radiance_version = rad.version if rad is not None else None

    case_dir = out_dir / "option1" / result.case
    case_dir.mkdir(parents=True, exist_ok=True)
    results_path = case_dir / "results.json"
    payload = _results_json(result, radiance_version, prepared.groups)
    payload["inputs"] = inputs
    results_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    _write_points(case_dir / "points.csv", result)

    _print_summary(result, results_path, out=out)

    return results_path


def _progress_printer(out):
    """A ``progress`` callback printing ``progress: <text>`` to ``out`` at once."""
    def printer(text: str) -> None:
        print(f"{PROGRESS_PREFIX}{text}", file=out, flush=True)
    return printer


def _positive_int(text: str) -> int:
    """argparse type for ``--processes``: an integer of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer of at least 1, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer of at least 1, got {text!r}")
    return value


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.run",
        description=(
            "Run LEED v4.1 BD+C EQ Credit Daylight, Option 1, on a project "
            "folder and write results.json and points.csv."
        ),
    )
    parser.add_argument(
        "folder", help="the project folder (project.yaml, materials.csv, spaces.csv)"
    )
    parser.add_argument(
        "--case", default=None,
        help="the case to run (default: the first case in spaces.csv)",
    )
    parser.add_argument(
        "--processes", type=_positive_int, default=1, metavar="N",
        help=(
            "run rfluxmtx/rcontrib with N processes (-n N); faster on a "
            "multi-core machine; same method, results equal within Monte "
            "Carlo noise (default: 1)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, RunError)
        with folder_lock(pathlib.Path(args.folder), "run"):
            run_folder(args.folder, args.case, processes=args.processes, progress=True)
    except (ValueError, RadianceError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
