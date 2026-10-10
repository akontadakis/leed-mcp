# tests/acceptance/test_acceptance_runs.py
"""``expected.json`` with one entry per run, and the per-run result file.

A fixture may hold several runs of the same building (acceptance spec 3:
run A with the earlier study's sensor points and trigger area, run B with
the tool's own defaults). ``expected.json`` names each run's project file,
targets and tolerances under ``runs``; ``LEED_ACCEPTANCE_RUN`` chooses one
per session, so a session never runs two by accident. A run without an
sDA or ASE target has no pass mark. Each run writes
``acceptance_result.json`` next to its outputs, the numbers the comparison
report needs.

Every test here is synthetic: it builds its own ``expected.json`` in
``tmp_path`` and hands ``_resolve_fixture`` a plain mapping as the
environment. Nothing sets either variable, builds a real fixture or runs
Radiance. The only literals are inside ``SYNTHETIC_`` constants.
"""
from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from . import conftest
from .conftest import (
    FIXTURE_ENV_VAR,
    RESULT_FILE,
    RUN_ENV_VAR,
    AcceptanceFixtureError,
    ResultFileWarning,
    _prepare_run,
    _resolve_fixture,
    _run_and_record,
    check_run_entry,
    expected_optional_count,
    expected_target,
    result_summary,
    run_option1_for_prepared,
    write_result_summary,
)

#: Two invented runs; numbers are placeholders, not a building's.
SYNTHETIC_RUNS = {
    "runs": {
        "A": {"project": "project_a.yaml", "option1": {"case": "base", "sda_pct": 50.0, "ase_pct": 20.0}},
        "B": {"project": "project_b.yaml", "option1": {"case": "base"}},
    }
}

#: Invented per-space thresholds, shaped as the credit rules hold them.
SYNTHETIC_RULES = {"option1": {"space_rule_sda_pct": [55.0], "ase": {"per_space_narrative_trigger_pct": 10.0}}}

#: An invented Option 1 result: two spaces, one group closed for two hours.
SYNTHETIC_RESULT = {
    "spaces": [
        {"room": "office a", "area_m2": 20.0, "points": 40, "sda_pct": 60.0, "ase_pct": 12.5},
        {"room": "office b", "area_m2": 10.0, "points": 20, "sda_pct": 50.0, "ase_pct": 5.0},
    ],
    "pcts": {"sda_area_pct": 56.7, "sda_point_pct": 56.7, "ase_area_pct": 10.0, "ase_point_pct": 10.0},
    "schedule": {"group_a": [True, False, False, True], "group_b": [True, True, True, True]},
    "closed_group_hours": 2,
    "spaces_sda_55": 1,
    "spaces_ase_over_10": 1,
    "points": 60,
    "area_m2": 30.0,
    "occupied_hours": 100,
    "warnings": ("group_a: an invented warning",),
}

#: A run entry with every key the gate reads, all invented.
SYNTHETIC_ENTRY = {
    "project": "project_a.yaml",
    "option1": {"case": "base", "aggregation": "floor_area", "sda_pct": 50.0, "ase_pct": 20.0},
    "tolerances": {"room_area_m2": 0.01, "glass_area_m2": 0.1, "frame_area_m2": 0.1},
}


def _fixture_folder(tmp_path, expected):
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8", newline="\n")
    return tmp_path


def test_a_fixture_with_runs_needs_the_run_named(tmp_path):
    root = _fixture_folder(tmp_path, SYNTHETIC_RUNS)

    with pytest.raises(AcceptanceFixtureError, match=(
        rf"expected\.json holds the runs A, B; set {RUN_ENV_VAR} to the one to run"
    )):
        _resolve_fixture({FIXTURE_ENV_VAR: str(root)})


def test_an_unknown_run_is_refused_naming_the_runs(tmp_path):
    root = _fixture_folder(tmp_path, SYNTHETIC_RUNS)

    with pytest.raises(AcceptanceFixtureError, match=rf"{RUN_ENV_VAR} is 'C'; expected\.json holds the runs A, B"):
        _resolve_fixture({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: "C"})


def test_the_chosen_run_gives_its_project_targets_and_output_folder(tmp_path):
    root = _fixture_folder(tmp_path, SYNTHETIC_RUNS)

    fixture = _resolve_fixture({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: "B"})

    assert fixture.run == "B"
    assert fixture.project_yaml == root / "project_b.yaml"
    assert fixture.out_dir == root / "_acceptance_run" / "B"
    assert fixture.expected == SYNTHETIC_RUNS["runs"]["B"]


def test_expected_json_without_runs_is_refused(tmp_path):
    root = _fixture_folder(tmp_path, {"option1": {"case": "base"}})

    with pytest.raises(AcceptanceFixtureError, match="expected.json: missing key 'runs'"):
        _resolve_fixture({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: "A"})


@pytest.mark.parametrize("project", ["/elsewhere/project.yaml", "../project.yaml", "", ".", "./"])
def test_a_run_project_outside_the_fixture_is_refused(tmp_path, project):
    root = _fixture_folder(tmp_path, {"runs": {"A": {"project": project}}})

    with pytest.raises(AcceptanceFixtureError, match=r"runs\.A\.project must name a file inside the fixture folder"):
        _resolve_fixture({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: "A"})


@pytest.mark.parametrize("run", [".", "..", "a/b", "a\\b", "/elsewhere", "C:x"])
def test_a_run_name_that_is_not_one_plain_folder_name_is_refused(tmp_path, run):
    root = _fixture_folder(tmp_path, {"runs": {run: {"project": "project_a.yaml"}}})

    with pytest.raises(AcceptanceFixtureError, match=rf"{RUN_ENV_VAR} .*: a run name must be one plain folder name"):
        _resolve_fixture({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: run})


def test_without_the_fixture_variable_it_still_skips(tmp_path):
    with pytest.raises(pytest.skip.Exception, match=FIXTURE_ENV_VAR):
        _resolve_fixture({RUN_ENV_VAR: "A"})


def test_a_run_without_a_target_has_no_pass_mark():
    run_a, run_b = SYNTHETIC_RUNS["runs"]["A"], SYNTHETIC_RUNS["runs"]["B"]

    assert expected_target(run_a, "option1.sda_pct") == run_a["option1"]["sda_pct"]
    assert expected_target(run_b, "option1.sda_pct") is None


def test_a_target_of_the_wrong_type_is_still_a_named_failure():
    with pytest.raises(AcceptanceFixtureError, match=r"option1\.sda_pct"):
        expected_target({"option1": {"sda_pct": "high"}}, "option1.sda_pct")


def test_an_optional_count_is_none_when_absent_and_checked_when_present():
    assert expected_optional_count({"prepared": {}}, "prepared.group_count") is None
    assert expected_optional_count({"prepared": {"group_count": 1}}, "prepared.group_count") == 1
    with pytest.raises(AcceptanceFixtureError, match=r"prepared\.group_count"):
        expected_optional_count({"prepared": {"group_count": True}}, "prepared.group_count")


def _result():
    data = SYNTHETIC_RESULT
    return types.SimpleNamespace(
        spaces=tuple(types.SimpleNamespace(**s) for s in data["spaces"]),
        blind_schedule={k: np.array(v) for k, v in data["schedule"].items()},
        occupied_hours=data["occupied_hours"],
        warnings=data["warnings"],
        **data["pcts"],
    )


def test_the_result_summary_holds_what_the_report_compares():
    summary = result_summary("A", _result(), SYNTHETIC_RULES)

    data = SYNTHETIC_RESULT
    assert summary["run"] == "A"
    assert summary["option1"] == data["pcts"]
    assert summary["points"] == data["points"]
    assert summary["spaces"] == len(data["spaces"])
    assert summary["area_m2"] == data["area_m2"]
    assert summary["spaces_sda_55"] == data["spaces_sda_55"]
    assert summary["spaces_ase_over_10"] == data["spaces_ase_over_10"]
    assert summary["blind_closed_group_hours"] == data["closed_group_hours"]
    assert summary["by_space"] == data["spaces"]
    assert summary["space_rule_sda_pct"] == SYNTHETIC_RULES["option1"]["space_rule_sda_pct"][0]
    assert (summary["occupied_hours"], summary["warnings"]) == (data["occupied_hours"], list(data["warnings"]))


def test_without_a_per_space_sda_band_the_summary_says_so():
    rules = {"option1": {"ase": SYNTHETIC_RULES["option1"]["ase"]}}

    summary = result_summary("A", _result(), rules)

    assert summary["space_rule_sda_pct"] is None
    assert summary["spaces_sda_55"] is None


def test_the_result_summary_is_written_as_json(tmp_path):
    path = write_result_summary(tmp_path / "out", "A", _result(), SYNTHETIC_RULES)

    assert path == tmp_path / "out" / RESULT_FILE
    assert json.loads(path.read_text(encoding="utf-8")) == result_summary("A", _result(), SYNTHETIC_RULES)


def test_the_run_puts_the_scene_folders_on_raypath(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(conftest, "load_rules", lambda path: {})
    monkeypatch.setattr(conftest, "find_binaries", lambda: None)
    monkeypatch.setattr(conftest, "subprocess_runner", lambda search_dirs=(): seen.setdefault("dirs", search_dirs))
    monkeypatch.setattr(conftest, "Radiance", lambda runner, log: ("radiance", runner, log))
    monkeypatch.setattr(conftest, "run_option1", lambda rad, project, prepared, spaces, rules, case: rad)
    scene_dir = tmp_path / "scene"
    project = types.SimpleNamespace(
        credit_rules_path=tmp_path / "rules.yaml",
        scene=types.SimpleNamespace(files=(scene_dir / "a.rad", scene_dir / "b.rad")),
        performance=None,
    )
    prepared = types.SimpleNamespace(root=tmp_path / "out")
    fixture = types.SimpleNamespace(expected={"option1": {"case": "base"}})

    _, _, log = run_option1_for_prepared(fixture, project, prepared, [])

    assert seen["dirs"] == (pathlib.Path(scene_dir).resolve(), (tmp_path / "out").resolve())
    assert log == tmp_path / "out" / "radiance.log"


def _chain(tmp_path):
    fixture = types.SimpleNamespace(out_dir=tmp_path / "out", run="A", expected=SYNTHETIC_ENTRY)
    project = types.SimpleNamespace(credit_rules_path=tmp_path / "rules.yaml")
    return fixture, project, None, None, [], None


def _fake_run(monkeypatch):
    result = _result()
    monkeypatch.setattr(conftest, "run_option1_for_prepared", lambda fixture, project, prepared, spaces: result)
    monkeypatch.setattr(conftest, "load_rules", lambda path: SYNTHETIC_RULES)
    return result


def test_the_run_writes_its_result_file(tmp_path, monkeypatch):
    result = _fake_run(monkeypatch)

    fixture, got = _run_and_record(_chain(tmp_path))

    assert got is result
    written = json.loads((tmp_path / "out" / RESULT_FILE).read_text(encoding="utf-8"))
    assert written["run"] == "A"


def test_a_failing_result_file_write_still_returns_the_result(tmp_path, monkeypatch):
    result = _fake_run(monkeypatch)

    def refuse(*args):
        raise OSError("disk full (invented)")

    monkeypatch.setattr(conftest, "write_result_summary", refuse)

    with pytest.warns(ResultFileWarning, match=rf"{RESULT_FILE} .*not written.*OSError: disk full \(invented\)"):
        fixture, got = _run_and_record(_chain(tmp_path))

    assert got is result


def test_a_complete_run_entry_passes_the_early_check():
    check_run_entry(SYNTHETIC_ENTRY)


@pytest.mark.parametrize(("section", "key", "value", "named"), [
    ("option1", "aggregation", "rooms", r"option1\.aggregation"),
    ("option1", "sda_pct", "high", r"option1\.sda_pct"),
    ("option1", "ase_pct", None, r"option1\.ase_pct"),
    ("tolerances", "glass_area_m2", "wide", r"tolerances\.glass_area_m2"),
    ("tolerances", "frame_area_m2", None, r"tolerances\.frame_area_m2"),
    ("tolerances", "room_area_m2", True, r"tolerances\.room_area_m2"),
])
def test_a_malformed_run_entry_fails_before_prepare(tmp_path, monkeypatch, section, key, value, named):
    entry = json.loads(json.dumps(SYNTHETIC_ENTRY))
    entry[section][key] = value
    root = _fixture_folder(tmp_path, {"runs": {"A": entry}})

    def never(*args):
        raise AssertionError("the run entry must be checked before the project is loaded")

    monkeypatch.setattr(conftest, "load_project", never)
    monkeypatch.setattr(conftest, "prepare", never)

    with pytest.raises(AcceptanceFixtureError, match=named):
        _prepare_run({FIXTURE_ENV_VAR: str(root), RUN_ENV_VAR: "A"})
