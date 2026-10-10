# tests/test_option2_command.py
"""Tests for the option2 command with a FAKE engine: no Radiance binary runs."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

import leed_daylight.drivers.option2 as option2_module
from option2_folder import ROOM, build_option2_project

LABELS = ["03-28_0900", "03-28_1500", "09-14_0900", "09-14_1500"]


class FakeEngine:
    def __init__(self, lux=500.0):
        self.lux = lux
        self.n_points = None

    def illuminance(self, moment):
        return np.full(self.n_points, self.lux)


def _engine_for(folder):
    from option2_folder import load_chain

    _, prepared, _, _, _ = load_chain(folder)
    engine = FakeEngine()
    engine.n_points = sum(int(g.points.shape[0]) for g in prepared.grids)
    return engine


def _folder(tmp_path, write_example, write_wea, **kwargs):
    return build_option2_project(tmp_path, write_example, write_wea, **kwargs)


def test_the_results_file_holds_every_field(tmp_path, write_example, write_wea):
    folder = _folder(tmp_path, write_example, write_wea)
    path = option2_module.option2_folder(folder, engine=_engine_for(folder), out=io.StringIO())
    assert path == folder / "_run" / "option2" / "base_case" / "option2_results.json"
    payload = json.loads(path.read_text())
    assert set(payload) == {
        "tool_version", "radiance_version", "case", "settings", "days", "skies",
        "spaces", "area_pct", "point_pct", "score", "warnings", "inputs",
    }
    assert payload["case"] == "base_case"
    assert [d["chosen"]["date"] for d in payload["days"]] == ["03-28", "09-14"]
    assert payload["days"][0]["equinox"] == "03-21"
    assert [s["label"] for s in payload["skies"]] == LABELS
    assert payload["skies"][0]["direct_normal_w_m2"] == 700.0
    (space,) = payload["spaces"]
    assert space["room"] == ROOM
    assert space["pass_pct"] == pytest.approx(100.0)
    assert payload["score"]["points"] == 3
    assert payload["settings"]["time_basis"] == "weather_file_local"


def test_points_csv_has_one_row_per_point(tmp_path, write_example, write_wea):
    folder = _folder(tmp_path, write_example, write_wea)
    engine = _engine_for(folder)
    option2_module.option2_folder(folder, engine=engine, out=io.StringIO())
    with open(folder / "_run" / "option2" / "base_case" / "points.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "room", "x", "y", "z", *(f"lux_{label}" for label in LABELS),
        "average_09:00", "average_15:00", "passes",
    ]
    assert len(rows) - 1 == engine.n_points
    assert rows[1][0] == ROOM and rows[1][-1] == "yes"


def test_the_summary_names_the_days_the_time_basis_and_the_room(tmp_path, write_example, write_wea):
    folder = _folder(tmp_path, write_example, write_wea)
    out = io.StringIO()
    option2_module.option2_folder(folder, engine=_engine_for(folder), out=out)
    printed = out.getvalue()
    assert "case: base_case" in printed
    assert "clearest days: 03-28" in printed and "09-14" in printed
    assert "time basis: weather file local time, no daylight saving" in printed
    assert "passing floor area: 100.0%" in printed
    assert "score: 3 points" in printed
    assert f"{ROOM}: 24.0 m2, 100.0% pass (0 low, 0 high)" in printed


def test_an_unready_project_exits_1_with_one_error_line(
    tmp_path, write_example, write_wea, capsys, monkeypatch,
):
    def unverified(d):
        d["option2"]["time_basis"] = "UNVERIFIED"

    folder = _folder(tmp_path, write_example, write_wea, mutate=unverified)
    # No Radiance on PATH: the settings refusal must still be the one shown.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert option2_module.main([str(folder)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "option2.time_basis" in err
    assert not (folder / "_run").exists()


def test_a_time_zone_mismatch_is_refused_before_anything_is_prepared(
    tmp_path, write_example, write_wea, capsys, monkeypatch,
):
    from option2_folder import sunny_wea_lines

    folder = _folder(tmp_path, write_example, write_wea)
    write_wea(
        folder / "weather" / "example.wea", latitude=42.0, longitude_W=80.0,
        time_zone=60.0, hours=sunny_wea_lines(),
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert option2_module.main([str(folder)]) == 1
    assert "time_zone" in capsys.readouterr().err
    assert not (folder / "_run").exists()


def test_an_unknown_case_exits_1(tmp_path, write_example, write_wea, capsys):
    folder = _folder(tmp_path, write_example, write_wea)
    assert option2_module.main([str(folder), "--case", "no_such_case"]) == 1
    assert "no_such_case" in capsys.readouterr().err


def test_the_command_needs_gendaylit():
    assert option2_module.OPTION2_BINARIES == ("oconv", "rtrace", "gendaylit", "xform")


def test_a_weather_file_with_no_clear_day_is_refused_before_anything_is_prepared(
    tmp_path, write_example, write_wea, capsys, monkeypatch,
):
    from option2_folder import sunny_wea_lines

    folder = _folder(tmp_path, write_example, write_wea, wea_lines=sunny_wea_lines(clear_days={}))
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert option2_module.main([str(folder)]) == 1
    assert "direct sun" in capsys.readouterr().err
    assert not (folder / "_run").exists()


def test_a_sun_below_the_horizon_is_refused_before_anything_is_prepared(
    tmp_path, write_example, write_wea, capsys, monkeypatch,
):
    def early(d):
        d["option2"]["times"] = ["05:00", "15:00"]

    folder = _folder(tmp_path, write_example, write_wea, mutate=early)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert option2_module.main([str(folder)]) == 1
    assert "below the horizon" in capsys.readouterr().err
    assert not (folder / "_run").exists()


def test_the_results_record_each_chosen_day_s_hourly_irradiance(tmp_path, write_example, write_wea):
    folder = _folder(tmp_path, write_example, write_wea)
    path = option2_module.option2_folder(folder, engine=_engine_for(folder), out=io.StringIO())
    payload = json.loads(path.read_text())
    # sunny_wea_lines: 700 W/m2 direct from 07:00 to 17:00, 100 W/m2 diffuse from 06:00 to 18:00.
    direct = [0.0] * 7 + [700.0] * 10 + [0.0] * 7
    diffuse = [0.0] * 6 + [100.0] * 12 + [0.0] * 6
    for day in payload["days"]:
        assert day["hourly_direct_normal_w_m2"] == direct
        assert day["hourly_diffuse_horizontal_w_m2"] == diffuse


def test_a_missing_radiance_is_reported_before_anything_is_prepared(
    tmp_path, write_example, write_wea, capsys, monkeypatch,
):
    folder = _folder(tmp_path, write_example, write_wea)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert option2_module.main([str(folder)]) == 1
    assert "Radiance binaries not found" in capsys.readouterr().err
    assert not (folder / "_run").exists()


def test_the_results_file_records_the_input_fingerprints(tmp_path, write_example, write_wea):
    from leed_daylight.project.config import load_project
    from leed_daylight.project.inputs import input_fingerprints

    folder = _folder(tmp_path, write_example, write_wea)
    path = option2_module.option2_folder(folder, engine=_engine_for(folder), out=io.StringIO())

    payload = json.loads(path.read_text())
    assert payload["inputs"] == input_fingerprints(load_project(folder / "project.yaml"), folder)
