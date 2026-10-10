# tests/acceptance/conftest.py
"""Resolve the private acceptance fixture, and run the pipeline against it.

This module names no fixture: not its path, not a room, not a building, not
a number. Every one of those lives outside this repository, in a folder the
user points at with the environment variable named below (spec 12, Global
Constraint 5, Global Constraint 9 -- the user decides when the simulation
runs, never this test).

**The fixture folder** (a real folder outside the repository; nothing about
it is committed): the folder the environment variable names holds

- one ``project.yaml``-style file per run (named in ``expected.json``), each
  an ordinary project file, as ``project.config`` reads;
- ``materials.csv``  -- an ordinary materials file, as ``project.materials`` reads;
- ``spaces.csv``     -- an ordinary spaces file, as ``project.spaces`` reads;
- ``expected.json``  -- the recorded targets of each run::

    {"runs": {
      "<run>": {
        "project": "<project file in the fixture folder>",
        "option1": {"case": "<case name>", "aggregation": "floor_area" | "points",
                    "sda_pct": <number, optional>, "ase_pct": <number, optional>},
        "rooms": {"count": <int>, "areas_m2": [<number>, ...]},
        "prepared": {"glass_area_m2": <number>, "frame_area_m2": <number>,
                     "group_count": <int, optional>, "grid_point_count": <int, optional>},
        "tolerances": {"room_area_m2": <number>, "glass_area_m2": <number>,
                        "frame_area_m2": <number>}
      }, ...}}

``rooms.areas_m2`` is sorted ascending: the floor polygons the fixture's
builder wrote. A run without ``sda_pct``/``ase_pct`` has no pass mark; an
absent count is not checked. ``LEED_ACCEPTANCE_RUN`` (``RUN_ENV_VAR``) names
the run of this session; outputs go to ``<fixture>/_acceptance_run/<run>/``,
with the run's ``acceptance_result.json`` (``RESULT_FILE``).

When the fixture variable is unset, or the folder it names has no
``expected.json``, every test that needs the fixture skips, naming what was
missing. Once ``expected.json`` exists, a missing run choice, an unknown
run, or a missing or wrong-typed key is a named failure, not a skip -- a
malformed fixture is a fixture bug, not "no fixture".

The pipeline chain this module runs, in order (R-P44): ``load_project`` ->
``read_scene`` -> ``load_materials`` -> ``validate_scene(...).raise_if_errors()``
**before** ``detect_rooms`` (a malformed floor polygon that validation
refuses by name must never reach room detection) -> ``load_spaces`` ->
``prepare`` -> ``load_rules`` -> ``find_binaries`` -> ``Radiance`` ->
``run_option1``. The output folder is ``<fixture root>/_acceptance_run/<run>/``,
outside the repository and persistent, so a stopped run can resume.

The chain is split into two session-scoped fixtures so one expensive run
feeds every test that needs it: ``_prepared`` stops after ``prepare()`` (the
room-detection and prepared-model checks need nothing past that), and
``_option1_run`` continues on to the actual Option 1 run (the sDA/ASE
checks, and the run's result file) by calling ``run_option1_for_prepared`` below. That function is the
ONE copy of the ``load_rules`` -> ``find_binaries`` -> ``Radiance`` ->
``run_option1`` tail: ``test_trace_scope_equivalence.py`` calls it too, on
the same prepared model, with ``performance.trace_scope`` overridden, so
there is no second hand-copy of this chain to drift out of sync. Nothing
under this package ever sets the environment variable, creates the fixture
folder, or executes a Radiance binary -- the user does.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import warnings
from collections.abc import Mapping

import numpy as np

import pytest

from leed_daylight import geometry
from leed_daylight.build.prepare import prepare
from leed_daylight.drivers.run import _raypath_dirs
from leed_daylight.drivers.run_option1 import run_option1
from leed_daylight.engine.radiance import Radiance, find_binaries, subprocess_runner
from leed_daylight.project.config import load_project
from leed_daylight.project.materials import load_materials
from leed_daylight.project.rules import load_rules
from leed_daylight.project.spaces import load_spaces
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms
from leed_daylight.scene.validate import validate_scene

#: The only things named in this file: the variables, not what they point to.
FIXTURE_ENV_VAR = "LEED_ACCEPTANCE_FIXTURE"
RUN_ENV_VAR = "LEED_ACCEPTANCE_RUN"
#: The per-run result file, in the run's output folder.
RESULT_FILE = "acceptance_result.json"


class AcceptanceFixtureError(ValueError):
    """``expected.json`` exists but is missing a key or has the wrong type."""


class ResultFileWarning(UserWarning):
    """The run's ``acceptance_result.json`` could not be written; the gate
    tests still judge the run."""


@dataclasses.dataclass(frozen=True)
class AcceptanceFixture:
    root: pathlib.Path  # OUTSIDE the repository, always
    project_yaml: pathlib.Path
    #: The chosen run's entry of ``expected.json`` (``runs.<run>``).
    expected: dict
    run: str
    #: ``<root>/_acceptance_run/<run>``.
    out_dir: pathlib.Path


def _resolve_fixture(environ: Mapping[str, str] | None = None) -> AcceptanceFixture:
    """The fixture and its chosen run, or ``pytest.skip`` naming what is missing.

    Never a silent default: an unset variable or a missing ``expected.json``
    is a skip that names what was checked, so a person reading a skipped
    run's output can act on it without reading this file. ``environ``
    defaults to ``os.environ``; tests hand in a plain mapping.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(FIXTURE_ENV_VAR)
    if not raw:
        pytest.skip(
            f"{FIXTURE_ENV_VAR} is not set; the acceptance suite needs a "
            "private fixture folder outside the repository"
        )
    root = pathlib.Path(raw)
    expected_path = root / "expected.json"
    if not expected_path.exists():
        pytest.skip(
            f"{root}: no expected.json found; the acceptance suite needs a "
            "private fixture folder outside the repository"
        )
    expected = json.loads(expected_path.read_text())
    if not isinstance(expected, dict):
        raise AcceptanceFixtureError("expected.json: expected a mapping at the top level")
    runs = expected_field(expected, "runs")
    if not isinstance(runs, dict) or not runs:
        raise AcceptanceFixtureError("expected.json: 'runs' must be a mapping of run name to its targets")
    names = ", ".join(runs)
    run = environ.get(RUN_ENV_VAR)
    if not run:
        raise AcceptanceFixtureError(
            f"expected.json holds the runs {names}; set {RUN_ENV_VAR} to the one to run"
        )
    if run not in runs:
        raise AcceptanceFixtureError(f"{RUN_ENV_VAR} is {run!r}; expected.json holds the runs {names}")
    _check_run_name(run)
    entry = runs[run]
    if not isinstance(entry, dict):
        raise AcceptanceFixtureError(f"expected.json: 'runs.{run}' must be a mapping")
    project = expected_str(entry, "project")
    relative = pathlib.PurePosixPath(project)
    if (
        relative.is_absolute() or ".." in relative.parts or pathlib.PureWindowsPath(project).drive
        or relative.name in ("", ".")
    ):
        raise AcceptanceFixtureError(
            f"expected.json: runs.{run}.project must name a file inside the fixture folder, got {project!r}"
        )
    return AcceptanceFixture(
        root=root, project_yaml=root / project, expected=entry, run=run,
        out_dir=root / "_acceptance_run" / run,
    )


def _check_run_name(run: str) -> None:
    """The run name becomes ``_acceptance_run/<run>``: it must be one plain
    folder name, never empty, ``.``, ``..``, a path with ``/`` or ``\\``,
    or an absolute or drive path that would put the outputs elsewhere."""
    if (
        run in ("", ".", "..") or "/" in run or "\\" in run
        or pathlib.PurePosixPath(run).is_absolute() or pathlib.PureWindowsPath(run).drive
    ):
        raise AcceptanceFixtureError(f"{RUN_ENV_VAR} {run!r}: a run name must be one plain folder name")


def expected_field(expected: dict, dotted_path: str):
    """A required field from ``expected.json``, present but untyped.

    ``dotted_path`` like ``"option1.sda_pct"``. A missing key or a
    non-mapping along the way is ``AcceptanceFixtureError``, naming the
    dotted path up to and including the missing part -- never a raw
    ``KeyError`` (Controller ruling: malformed ``expected.json`` is a
    failure naming the key, not a skip). Presence only: callers that need a
    particular type should use ``expected_number``, ``expected_count``,
    ``expected_str``, ``expected_enum`` or ``expected_number_list`` instead,
    so a wrong-typed value is also a named failure rather than a raw
    ``TypeError`` three lines later.
    """
    node = expected
    parts = dotted_path.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            seen = ".".join(parts[: i + 1])
            raise AcceptanceFixtureError(f"expected.json: missing key {seen!r}")
        node = node[part]
    return node


def _type_name(value) -> str:
    return type(value).__name__


def expected_number(expected: dict, dotted_path: str) -> float:
    """A required, finite, non-boolean number from ``expected.json``."""
    value = expected_field(expected, dotted_path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be a number, got "
            f"{_type_name(value)} {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be a finite number, got {value!r}"
        )
    return result


def expected_count(expected: dict, dotted_path: str) -> int:
    """A required non-negative integer from ``expected.json`` (a count):
    ``bool`` is rejected even though it is technically an ``int``."""
    value = expected_field(expected, dotted_path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be an integer, got "
            f"{_type_name(value)} {value!r}"
        )
    if value < 0:
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be >= 0, got {value!r}"
        )
    return value


def _last_key_present(expected: dict, dotted_path: str) -> bool:
    """Whether the last key of ``dotted_path`` is present; every key before
    it is required (``expected_field``)."""
    *parents, last = dotted_path.split(".")
    node = expected_field(expected, ".".join(parents)) if parents else expected
    return isinstance(node, dict) and last in node


def expected_optional_count(expected: dict, dotted_path: str) -> int | None:
    """``expected_count``, or None when the last key is absent (a count the
    run does not check, such as a window-group count the tool forms its
    own way)."""
    return expected_count(expected, dotted_path) if _last_key_present(expected, dotted_path) else None


def expected_target(expected: dict, dotted_path: str) -> float | None:
    """``expected_number``, or None when the last key is absent: a run with
    no target there has no pass mark (acceptance spec 3, run B)."""
    return expected_number(expected, dotted_path) if _last_key_present(expected, dotted_path) else None


def expected_str(expected: dict, dotted_path: str) -> str:
    """A required string from ``expected.json``."""
    value = expected_field(expected, dotted_path)
    if not isinstance(value, str):
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be a string, got "
            f"{_type_name(value)} {value!r}"
        )
    return value


def expected_enum(expected: dict, dotted_path: str, permitted: frozenset) -> str:
    """A required string from ``expected.json``, restricted to ``permitted``."""
    value = expected_str(expected, dotted_path)
    if value not in permitted:
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be one of "
            f"{sorted(permitted)}, got {value!r}"
        )
    return value


def expected_number_list(expected: dict, dotted_path: str) -> list[float]:
    """A required list of finite, non-boolean numbers from ``expected.json``,
    each one checked and named individually (``dotted_path[i]``)."""
    value = expected_field(expected, dotted_path)
    if not isinstance(value, list):
        raise AcceptanceFixtureError(
            f"expected.json: {dotted_path!r} must be a list, got "
            f"{_type_name(value)} {value!r}"
        )
    result: list[float] = []
    for i, item in enumerate(value):
        item_path = f"{dotted_path}[{i}]"
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AcceptanceFixtureError(
                f"expected.json: {item_path!r} must be a number, got "
                f"{_type_name(item)} {item!r}"
            )
        as_float = float(item)
        if not math.isfinite(as_float):
            raise AcceptanceFixtureError(
                f"expected.json: {item_path!r} must be a finite number, got {item!r}"
            )
        result.append(as_float)
    return result


#: The two `option1.aggregation` values `run_option1` produces (project.config's
#: own `_AGGREGATIONS`, spelled out here rather than imported, so this
#: fixture-validation module owns its own vocabulary check independent of
#: any change to the loader's internals).
_AGGREGATIONS = frozenset({"floor_area", "points"})


def expected_aggregation(expected: dict) -> str:
    """``option1.aggregation``, restricted to ``floor_area``/``points`` -- a
    named failure for anything else, never a silent fall-through to
    whichever branch an ``if/else`` happens to take (fix round 1, item 4)."""
    return expected_enum(expected, "option1.aggregation", _AGGREGATIONS)


def check_run_entry(expected: dict) -> None:
    """Read, before ``prepare``, every key of the run entry that the gate
    tests read after the run: the case, the aggregation, both targets and
    the three tolerances. A malformed entry then fails in seconds, not
    after hours of Radiance."""
    expected_str(expected, "option1.case")
    expected_aggregation(expected)
    expected_target(expected, "option1.sda_pct")
    expected_target(expected, "option1.ase_pct")
    for key in ("room_area_m2", "glass_area_m2", "frame_area_m2"):
        expected_number(expected, f"tolerances.{key}")


def expected_rooms(expected: dict) -> tuple[int, list[float]]:
    """``rooms.count`` and ``rooms.areas_m2``, type-checked and cross-checked:
    the list's length must equal the count, or the mismatch is a named
    failure (fix round 1, item 3) rather than a silently truncated `zip`."""
    count = expected_count(expected, "rooms.count")
    areas = expected_number_list(expected, "rooms.areas_m2")
    if len(areas) != count:
        raise AcceptanceFixtureError(
            f"expected.json: 'rooms.areas_m2' has {len(areas)} entries, "
            f"but 'rooms.count' is {count}; they must match"
        )
    return count, areas


def frame_area_m2(prims_by_name: Mapping[str, object], openings: Mapping[str, dict]) -> float:
    """The prepared model's total frame area (M8): each exterior opening's
    own primitive area minus its achieved glass area, summed.

    A named :class:`AcceptanceFixtureError` -- never a raw ``KeyError`` --
    when an opening recorded in ``prepared.manifest['openings']`` has no
    matching primitive in the scene (``prims_by_name``); that mismatch is
    a fixture/prepared-model inconsistency, not a Python lookup bug, and
    deserves the same named-failure treatment as every other malformed
    ``expected.json``/prepared-model check in this module.
    """
    total = 0.0
    for name, record in openings.items():
        prim = prims_by_name.get(name)
        if prim is None:
            raise AcceptanceFixtureError(
                f"prepared.manifest['openings']: opening {name!r} has no "
                "matching primitive in the scene"
            )
        total += geometry.area(prim.points) - record["achieved_glass_area_m2"]
    return total


def result_summary(run: str, result, rules: Mapping) -> dict:
    """What the comparison report needs from one Option 1 run: the four
    project percentages, points, spaces and their area, spaces at or over
    the first per-space sDA band (``option1.space_rule_sda_pct[0]``), spaces
    over the ASE narrative trigger, closed blind group-hours, the occupied
    hours, the run's warnings and each space's own numbers. ``result`` is an
    ``Option1Result`` (or anything with its ``spaces``, ``blind_schedule``,
    ``occupied_hours``, ``warnings`` and percentage fields).

    ``space_rule_sda_pct`` is optional in the credit rules: without it the
    summary records ``space_rule_sda_pct: null`` and ``spaces_sda_55: null``."""
    band = rules["option1"].get("space_rule_sda_pct")
    sda_space_pct = band[0] if band else None
    ase_space_pct = rules["option1"]["ase"]["per_space_narrative_trigger_pct"]
    spaces = [
        {"room": s.room, "area_m2": s.area_m2, "points": s.points, "sda_pct": s.sda_pct, "ase_pct": s.ase_pct}
        for s in result.spaces
    ]
    return {
        "run": run,
        "option1": {
            "sda_area_pct": result.sda_area_pct, "sda_point_pct": result.sda_point_pct,
            "ase_area_pct": result.ase_area_pct, "ase_point_pct": result.ase_point_pct,
        },
        "points": sum(s["points"] for s in spaces),
        "spaces": len(spaces),
        "area_m2": sum(s["area_m2"] for s in spaces),
        "space_rule_sda_pct": sda_space_pct,
        "spaces_sda_55": (
            None if sda_space_pct is None else sum(1 for s in spaces if s["sda_pct"] >= sda_space_pct)
        ),
        "spaces_ase_over_10": sum(1 for s in spaces if s["ase_pct"] > ase_space_pct),
        "blind_closed_group_hours": int(sum(int(np.count_nonzero(~np.asarray(open_, dtype=bool)))
                                            for open_ in result.blind_schedule.values())),
        "occupied_hours": result.occupied_hours,
        "warnings": list(result.warnings),
        "by_space": spaces,
    }


def write_result_summary(out_dir: pathlib.Path, run: str, result, rules: Mapping) -> pathlib.Path:
    """Write ``result_summary`` as ``<out_dir>/acceptance_result.json``."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(result_summary(run, result, rules), indent=1) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def acceptance_fixture() -> AcceptanceFixture:
    """Reads ``LEED_ACCEPTANCE_FIXTURE`` and ``LEED_ACCEPTANCE_RUN``. Skips,
    naming the variable, when the first is unset or the folder has no
    ``expected.json``."""
    return _resolve_fixture()


@pytest.fixture(scope="session")
def _prepared():
    """The chain through ``prepare()``, once per test session.

    Returns ``(fixture, project, prims, rooms, spaces, prepared)``. A
    fixture-resolving function is called directly here, not the
    function-scoped ``acceptance_fixture`` fixture: a session-scoped fixture
    cannot depend on a function-scoped one, and calling the plain function
    gets the same skip-with-message behaviour either way.
    """
    return _prepare_run()


def _prepare_run(environ: Mapping[str, str] | None = None):
    """``_prepared``'s chain as a plain function (``environ`` as for
    ``_resolve_fixture``): the run entry is checked first
    (``check_run_entry``), then the chain runs through ``prepare()``."""
    fixture = _resolve_fixture(environ)
    check_run_entry(fixture.expected)
    project = load_project(fixture.project_yaml)
    prims = read_scene(project.scene.files, project.scene.units)
    roles = load_materials(fixture.root / "materials.csv")
    validate_scene(prims, project.scene.units, roles).raise_if_errors()
    rooms = detect_rooms(prims, roles)
    spaces = load_spaces(fixture.root / "spaces.csv")
    case = expected_str(fixture.expected, "option1.case")
    prepared = prepare(project, prims, roles, rooms, spaces, case, fixture.out_dir)
    return fixture, project, prims, rooms, spaces, prepared


def _project_with_trace_scope(project, trace_scope: str | None):
    """``project`` unchanged when ``trace_scope`` is ``None``; otherwise a
    ``dataclasses.replace``d copy with only ``performance.trace_scope``
    swapped -- every other field, including ``performance.linked_zones``,
    passes through unchanged. Pure: no I/O, no fixture, so it is testable
    directly (task 4.8 review, finding 1) without
    ``LEED_ACCEPTANCE_FIXTURE``: the one place this module's ``trace_scope``
    override happens, guarded by its own test in
    ``tests/acceptance/test_trace_scope_equivalence.py`` so a future change
    here cannot silently start mutating something else.
    """
    if trace_scope is None:
        return project
    return dataclasses.replace(
        project, performance=dataclasses.replace(project.performance, trace_scope=trace_scope),
    )


#: The credit rules read in this test session, per file and reader (``_rules``).
_RULES_READ: dict[tuple[pathlib.Path, object], dict] = {}


def _rules(path: pathlib.Path) -> dict:
    """``load_rules(path)``, read once per test session: the Option 1 run and
    its result file share one read of the credit rules. The key holds the
    reader as well, so a test that replaces ``load_rules`` gets its own."""
    key = (pathlib.Path(path), load_rules)
    if key not in _RULES_READ:
        _RULES_READ[key] = load_rules(path)
    return _RULES_READ[key]


def run_option1_for_prepared(
    fixture: AcceptanceFixture, project, prepared, spaces,
    *, trace_scope: str | None = None, log_name: str = "radiance.log",
):
    """The chain from ``load_rules`` through ``run_option1``, on an
    already-``prepare``d model -- the ONE copy of this tail, so both
    ``_option1_run`` below (the project's own scope) and
    ``tests/acceptance/test_trace_scope_equivalence.py`` (both scopes, via
    ``trace_scope``) call the same chain instead of each keeping its own
    hand-copy that could silently drift apart (task 4.8 review, finding 1:
    "if one changes, the other silently keeps the old chain").

    ``trace_scope``, when given, replaces ``project.performance.trace_scope``
    -- via ``_project_with_trace_scope``, never any other substitution --
    before the run. ``log_name`` names the Radiance log file under
    ``prepared.root``; callers that run this more than once on the same
    prepared model (the trace_scope equivalence tests do, once per scope)
    must give each call its own ``log_name`` so neither run's log
    overwrites the other's.

    The runner puts each scene file's folder and the prepared model folder
    on ``RAYPATH``, as the ``run`` command does (``drivers.run._raypath_dirs``),
    so a scene's relative ``.xml``/``.cal`` reference (a BSDF next to the
    scene) resolves.
    """
    rules = _rules(project.credit_rules_path)
    find_binaries()
    log_path = prepared.root / log_name
    rad = Radiance(subprocess_runner(search_dirs=_raypath_dirs(project, prepared.root)), log_path)
    scoped_project = _project_with_trace_scope(project, trace_scope)
    case = expected_str(fixture.expected, "option1.case")
    return run_option1(rad, scoped_project, prepared, spaces, rules, case)


@pytest.fixture(scope="session")
def _option1_run(_prepared):
    """The full chain through ``run_option1()``, once per test session; the
    run's ``acceptance_result.json`` is written beside its outputs."""
    return _run_and_record(_prepared)


def _run_and_record(prepared_chain):
    """``_option1_run`` as a plain function: run Option 1 on the prepared
    chain, write the result file, return ``(fixture, result)``.

    The result file is written after hours of Radiance. A failure there
    (the rules, the summary or the disk) must not turn the gate tests into
    errors: it is a :class:`ResultFileWarning` naming the error and the
    file, and the result is still returned, so the sDA/ASE gate tests judge
    and print the run's numbers."""
    fixture, project, prims, rooms, spaces, prepared = prepared_chain
    result = run_option1_for_prepared(fixture, project, prepared, spaces)
    try:
        write_result_summary(fixture.out_dir, fixture.run, result, _rules(project.credit_rules_path))
    except Exception as exc:  # noqa: BLE001 -- any failure here must not hide the verdict
        warnings.warn(
            f"{RESULT_FILE} not written to {pathlib.Path(fixture.out_dir) / RESULT_FILE}: "
            f"{type(exc).__name__}: {exc}; the gate tests still judge the run",
            ResultFileWarning,
        )
    return fixture, result
