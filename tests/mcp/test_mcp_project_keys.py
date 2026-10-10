# tests/mcp/test_mcp_project_keys.py
"""The allowed-keys table against the loader (spec 6.3)."""
import pytest
import yaml

from leed_daylight.edit.project_yaml import render, set_path, template_keys
from leed_daylight.project.config import check_project_draft
from leed_mcp.adapters.project_keys import KEYS, check_keys, rule_for
from leed_mcp.errors import Refused

class Together(dict):
    """Several keys a sample sets together, because the loader needs them so."""


#: One invented, round value per key, or ``Together``. ``*`` (a storey) is
#: written ``L1``.
DRAWING = {"storey": "L1", "image": "plan.png",
           "points": [{"pixel": [0, 0], "model": [0, 0]}, {"pixel": [100, 0], "model": [10, 0]}]}
SAMPLES = {
    "project.name": "Example", "project.credit": "Example credit",
    "site.latitude_N": 40.0, "site.longitude_E": -80.0, "site.meridian_radiance": 75.0,
    "site.north_rotation_deg": 5.0, "site.north_rotation_source": "Example: site plan",
    "scene.units": "m", "scene.files": ["scene.rad"], "standard.daylight": "LM-83-12",
    "standard.credit_rules": "credit_rules.yaml", "weather.file": "weather/example.wea",
    "glazing.dirt_factor": 0.9, "glazing.frame_share": {"default": 0.2}, "glazing.frame_share.default": 0.2,
    "glazing.frame_reflectance": 0.5, "glazing.frame_style": "uniform_scale", "glazing.reveal_depth_m": 0.1,
    "glazing.reveal_reflectance": 0.5, "glazing.vt_includes_dirt": True,
    "glazing.dirt_factor.vertical": 0.9, "glazing.dirt_factor.sloped": 0.8, "glazing.dirt_factor.skylight": 0.7,
    "glazing.frame_share.sloped": 0.1, "glazing.frame_share.skylight": 0.1, "glazing.frame_share.glazing": 0.1,
    "glazing.frame_share.glazing_curtain_wall": 0.1, "glazing.frame_share.glazing_diffusing": 0.1,
    "glazing.skylight_well": "modelled",
    "glazing.skylight_well_reflectance": Together({"glazing.skylight_well": 0.5,
                                                   "glazing.skylight_well_reflectance": 0.5}),
    "reflectance.policy": "fill_gaps",
    **{f"reflectance.{role}": 0.5 for role in ("ceiling", "context", "fixed_shading", "floor", "furniture",
                                              "glazing", "glazing_curtain_wall", "glazing_diffusing",
                                              "glazing_interior", "ground", "ignore", "wall")},
    "grid.spacing_m": 0.5, "grid.work_plane_m": 0.8, "grid.min_clearance_m": 0.3, "grid.points_file": "points.csv",
    "analysis.hours": "09:00-17:00", "analysis.daylight_saving": "none", "analysis.aggregation": "points",
    "analysis.moveable_furniture": "include",
    "performance.trace_scope": "storey", "performance.linked_zones": {"L1": ["A", "B"]},
    "performance.linked_zones.*": ["A", "B"],
    "sky.basis": "Reinhart", "sky.MF": 2, "radiance.ambient": {"draft": "-ab 1", "full": "-ab 2"},
    "radiance.ambient.draft": "-ab 1", "radiance.ambient.full": "-ab 2", "radiance.direct": "-dj 0",
    "blinds.transmittance_diffuse": 0.1, "blinds.transmittance_specular": 0.0, "blinds.room_side_reflectance": 0.5,
    "blinds.trigger_area": "grid",
    "option2.equinox_dates": ["03-20", "09-20"], "option2.clearest_day_window_days": 10,
    "option2.times": ["10:00", "14:00"], "option2.sky": "weather_file", "option2.lux_min": 300,
    "option2.lux_max": 3000, "option2.blinds": "exclude", "option2.time_basis": "weather_file_local",
    "views": {"eye_height_m": 1.2, "eye_height_source": "Example: seated"},
    "views.eye_height_m": Together({"views.eye_height_m": 1.2, "views.eye_height_source": "Example: seated"}),
    "views.eye_height_source": Together({"views.eye_height_m": 1.2, "views.eye_height_source": "Example: seated"}),
    "views.max_interior_glass": Together({"views.eye_height_m": 1.2, "views.eye_height_source": "Example: seated",
                                          "views.max_interior_glass": 2}),
    "views.angular_step_deg": Together({"views.eye_height_m": 1.2, "views.eye_height_source": "Example: seated",
                                        "views.angular_step_deg": 1.0}),
    "report": {"renderer": "pillow"}, "report.renderer": "pillow", "report.deviations": ["Example deviation"],
    "report.prepared_by": "Example office",
    "atlas": {"drawings": [DRAWING]}, "atlas.drawings": [DRAWING],
    "calculator": {"units": "SI"}, "calculator.units": "SI",
}


def _setting(path):
    """The keys and values a sample sets for ``path``."""
    sample = SAMPLES[path]
    return sample if isinstance(sample, Together) else {path.replace("*", "L1"): sample}


def test_every_key_has_a_sample_and_every_sample_a_key():
    assert set(SAMPLES) == {key.path for key in KEYS}


def test_every_template_key_is_in_the_table():
    assert [k.path for k in template_keys() if rule_for(k.path) is None] == []


@pytest.mark.parametrize("path", [key.path for key in KEYS])
def test_the_loader_accepts_every_key_in_the_table(path, tmp_path, write_example):
    project = write_example(tmp_path)
    (project.parent / "points.csv").write_text("x,y,z\n0,0,0.8\n")
    (project.parent / "plan.png").write_bytes(b"")
    raw = yaml.safe_load(project.read_text())
    for key, value in _setting(path).items():
        set_path(raw, key, value)
    project.write_text(render(raw))
    errors, unfinished = check_project_draft(project)
    assert (errors, unfinished) == ((), ())


def test_the_loader_check_does_refuse_a_bad_value(tmp_path, write_example):
    project = write_example(tmp_path)
    raw = yaml.safe_load(project.read_text())
    set_path(raw, "glazing.frame_style", "bogus")
    project.write_text(render(raw))
    assert "glazing.frame_style" in check_project_draft(project)[0][0]


@pytest.mark.parametrize("values, unset, words", [
    ({"site.latitude": 1.0}, [], "'site.latitude' is not a project.yaml key"),
    ({"reflectance.celing": 0.5}, [], "'reflectance.celing' is not"),
    ({}, ["site.latitude_N"], "'site.latitude_N' is required"),
    ({"views.eye_height_m": 1.2}, ["views.eye_height_m"], "is both set and removed"),
    ({"views": {}, "views.eye_height_m": 1.2}, [], "its parent 'views'"),
    ({"performance.linked_zones.L1.5": ["A"]}, [], "is not a project.yaml key"),
])
def test_check_keys_refuses(values, unset, words):
    with pytest.raises(Refused) as caught:
        check_keys(values, unset)
    content = caught.value.result.structured_content
    assert content["advisories"][0]["code"] == "INPUT_INVALID" and words in content["reason"]


def test_check_keys_accepts_optional_removals_and_a_storey():
    check_keys({"performance.linked_zones.Level 2": ["A"]}, ["reflectance.ground", "views", "grid.points_file"])
