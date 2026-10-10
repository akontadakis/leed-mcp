# tests/test_project_checks.py
"""Task 3.2: the cross-field and filesystem checks on a loaded project.

Every test mutates a fresh copy of the example project (via the
``write_example``/``example_project_dir`` fixtures from conftest.py) and
loads it through ``load_project``, which wires ``check_project`` in at the
end (config.py). A check that refuses raises ``ProjectError`` naming the
key; a check that only warns lets the load succeed and the warning shows up
on ``project.warnings``.
"""
from __future__ import annotations

import pytest

from leed_daylight.constants import UNVERIFIED
from leed_daylight.metrics.occupancy import occupied_mask
from leed_daylight.project.checks import check_project
from leed_daylight.project.config import ProjectError, load_project


# ---------------------------------------------------------------------------
# The happy path and the never-raises contract
# ---------------------------------------------------------------------------

def test_a_valid_project_has_no_errors(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")
    # The example's option2 section is settled (time_basis
    # weather_file_local), so it carries no time_basis warning.
    errors, warnings = check_project(project)
    assert errors == ()
    assert not any("time_basis" in w for w in warnings)


def test_every_error_is_collected(write_example, tmp_path):
    def mutate(d):
        d["grid"]["spacing_m"] = 2.0
        d["sky"]["MF"] = 99
        d["glazing"]["dirt_factor"] = 5.0
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "3 problem(s)" in message
    assert "grid.spacing_m" in message
    assert "sky.MF" in message
    assert "glazing.dirt_factor" in message


# ---------------------------------------------------------------------------
# Filesystem checks
# ---------------------------------------------------------------------------

def test_a_missing_weather_file_is_refused(write_example, tmp_path):
    def mutate(d):
        d["weather"]["file"] = "weather/does_not_exist.wea"
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="weather.file"):
        load_project(path)


def test_a_missing_scene_file_is_refused(write_example, tmp_path):
    def mutate(d):
        d["scene"]["files"] = ["does_not_exist.rad"]
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=r"scene\.files\[0\]"):
        load_project(path)


def test_a_missing_credit_rules_file_is_refused(write_example, tmp_path):
    def mutate(d):
        d["standard"]["credit_rules"] = "does_not_exist.yaml"
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="standard.credit_rules"):
        load_project(path)


# ---------------------------------------------------------------------------
# glazing.frame_share: no client-derived default, no silent default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [None, 1.0, -0.1])
def test_frame_share_default_must_be_stated_and_in_range(write_example, tmp_path, bad_value):
    def mutate(d):
        d["glazing"]["frame_share"]["default"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=r"glazing\.frame_share\.default"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [None, 1.5])
def test_other_frame_share_values_must_be_stated_and_in_range(write_example, tmp_path, bad_value):
    def mutate(d):
        d["glazing"]["frame_share"]["glazing_curtain_wall"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=r"glazing\.frame_share\.glazing_curtain_wall"):
        load_project(path)


# ---------------------------------------------------------------------------
# glazing.dirt_factor: a reduction, never a gain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0.0, 1.5])
def test_dirt_factor_out_of_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["glazing"]["dirt_factor"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="glazing.dirt_factor"):
        load_project(path)


# ---------------------------------------------------------------------------
# reflectance.by_role and the ruling's added frame/reveal/blind rows: every
# one of these shares the "a reflectance or transmittance above 1 creates
# light" family, so they are folded into one parametrised range test.
# ---------------------------------------------------------------------------

def _set_reflectance_ceiling(d, value):
    d["reflectance"]["ceiling"] = value


def _set_reflectance_wall(d, value):
    d["reflectance"]["wall"] = value


def _set_frame_reflectance(d, value):
    d["glazing"]["frame_reflectance"] = value


def _set_reveal_reflectance(d, value):
    d["glazing"]["reveal_reflectance"] = value


def _set_room_side_reflectance(d, value):
    d["blinds"]["room_side_reflectance"] = value


def _set_transmittance_diffuse(d, value):
    d["blinds"]["transmittance_diffuse"] = value


def _set_transmittance_specular(d, value):
    d["blinds"]["transmittance_specular"] = value


RANGE_CASES = (
    ("reflectance.ceiling", _set_reflectance_ceiling, 1.5),
    ("reflectance.wall", _set_reflectance_wall, -0.1),
    ("glazing.frame_reflectance", _set_frame_reflectance, 1.5),
    ("glazing.reveal_reflectance", _set_reveal_reflectance, -0.1),
    ("blinds.room_side_reflectance", _set_room_side_reflectance, 1.5),
    ("blinds.transmittance_diffuse", _set_transmittance_diffuse, 1.5),
    ("blinds.transmittance_specular", _set_transmittance_specular, -0.1),
)


@pytest.mark.parametrize(
    "key, setter, bad_value", RANGE_CASES, ids=[c[0] for c in RANGE_CASES]
)
def test_a_reflectance_or_transmittance_out_of_range_is_refused(
    write_example, tmp_path, key, setter, bad_value
):
    def mutate(d):
        setter(d, bad_value)
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=key.replace(".", r"\.")):
        load_project(path)


def test_reveal_depth_negative_is_refused(write_example, tmp_path):
    def mutate(d):
        d["glazing"]["reveal_depth_m"] = -0.1
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match=r"glazing\.reveal_depth_m"):
        load_project(path)


def test_reveal_depth_of_zero_is_accepted(write_example, tmp_path):
    def mutate(d):
        d["glazing"]["reveal_depth_m"] = 0.0
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert project.glazing.reveal_depth_m == 0.0


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_grid_spacing_of_exactly_0_60_is_accepted(write_example, tmp_path):
    def mutate(d):
        d["grid"]["spacing_m"] = 0.60
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert project.grid.spacing_m == 0.60


def test_grid_spacing_above_0_60_is_refused(write_example, tmp_path):
    def mutate(d):
        d["grid"]["spacing_m"] = 0.61
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "grid.spacing_m" in message
    assert "600" in message


def test_grid_spacing_of_zero_is_refused(write_example, tmp_path):
    def mutate(d):
        d["grid"]["spacing_m"] = 0.0
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="grid.spacing_m"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [0.0, 2.0])
def test_grid_work_plane_out_of_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["grid"]["work_plane_m"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="grid.work_plane_m"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [-0.1, 2.0])
def test_grid_min_clearance_out_of_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["grid"]["min_clearance_m"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="grid.min_clearance_m"):
        load_project(path)


# ---------------------------------------------------------------------------
# blinds: a shade cannot transmit more than it receives
# ---------------------------------------------------------------------------

def test_blinds_transmittance_sum_exceeds_one_is_refused(write_example, tmp_path):
    def mutate(d):
        d["blinds"]["transmittance_diffuse"] = 0.60
        d["blinds"]["transmittance_specular"] = 0.60
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "blinds.transmittance_diffuse" in message
    assert "blinds.transmittance_specular" in message


# ---------------------------------------------------------------------------
# sky.MF: Reinhart subdivisions
# ---------------------------------------------------------------------------

def test_sky_mf_must_be_a_reinhart_subdivision(write_example, tmp_path):
    def mutate(d):
        d["sky"]["MF"] = 3
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "sky.MF" in message
    assert "[1, 2, 4, 6]" in message


# ---------------------------------------------------------------------------
# analysis.hours: what selects every reported hour
# ---------------------------------------------------------------------------

def test_analysis_hours_that_fails_to_parse_is_refused(write_example, tmp_path):
    def mutate(d):
        d["analysis"]["hours"] = "garbage"
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="analysis.hours"):
        load_project(path)


@pytest.mark.parametrize(
    "hours",
    [
        "08:00-18:00",
        "00:00-24:00",
        "8:00-18:00",
        "08:30-18:00",
        "18:00-08:00",
        "08:00-25:00",
        "08:00-08:00",
        "garbage",
        "02:00-10:00",
        "03:00-11:00",
    ],
)
@pytest.mark.parametrize("daylight_saving", ["none", "US", "EU"])
def test_the_hours_check_agrees_with_occupied_mask(write_example, tmp_path, hours, daylight_saving):
    def mutate(d):
        d["analysis"]["hours"] = hours
        d["analysis"]["daylight_saving"] = daylight_saving
        return d

    path = write_example(tmp_path, mutate=mutate)

    try:
        occupied_mask(hours, daylight_saving)
        mask_accepts = True
    except ValueError:
        mask_accepts = False

    if mask_accepts:
        project = load_project(path)
        assert project.analysis.hours == hours
    else:
        with pytest.raises(ProjectError, match="analysis.hours"):
            load_project(path)


# ---------------------------------------------------------------------------
# option2
# ---------------------------------------------------------------------------

def test_option2_lux_min_must_not_exceed_lux_max(write_example, tmp_path):
    def mutate(d):
        d["option2"]["lux_min"] = 4000
        d["option2"]["lux_max"] = 3000
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="option2.lux_min"):
        load_project(path)


def test_option2_time_basis_unverified_is_a_warning(write_example, tmp_path):
    def mutate(d):
        d["option2"]["time_basis"] = UNVERIFIED
        return d

    project = load_project(write_example(tmp_path, mutate=mutate))
    assert any("option2.time_basis" in w and UNVERIFIED in w for w in project.warnings)


def test_option2_blinds_unverified_is_a_warning(write_example, tmp_path):
    def mutate(d):
        d["option2"]["blinds"] = UNVERIFIED
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert any("option2.blinds" in w for w in project.warnings)


def test_option2_unverified_warning_never_becomes_an_error(write_example, tmp_path):
    def mutate(d):
        d["option2"]["blinds"] = UNVERIFIED
        d["option2"]["time_basis"] = UNVERIFIED
        return d

    path = write_example(tmp_path, mutate=mutate)
    # Must not raise: an UNVERIFIED Option 2 field is a warning, never a
    # refusal -- a project that only wants Option 1 must still load.
    project = load_project(path)
    assert any("option2.blinds" in w for w in project.warnings)
    assert any("option2.time_basis" in w for w in project.warnings)


# ---------------------------------------------------------------------------
# site.latitude_N: outside the band, the run continues but the report must
# say so
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("latitude", [30.0, 50.0])
def test_latitude_outside_band_is_a_warning(write_example, tmp_path, latitude):
    def mutate(d):
        d["site"]["latitude_N"] = latitude
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert any("site.latitude_N" in w for w in project.warnings)


def test_a_latitude_inside_the_band_produces_no_warning(example_project_dir):
    # The unmutated example is at 42.0 degrees N.
    project = load_project(example_project_dir / "project.yaml")
    assert not any("latitude" in w for w in project.warnings)


@pytest.mark.parametrize("latitude", [37.0, 48.0])
def test_latitude_band_boundaries_produce_no_warning(write_example, tmp_path, latitude):
    def mutate(d):
        d["site"]["latitude_N"] = latitude
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert not any("latitude" in w for w in project.warnings)


# ---------------------------------------------------------------------------
# I6: site fields outside their physically possible range are refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [-91.0, 91.0])
def test_latitude_outside_the_physical_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["site"]["latitude_N"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.latitude_N"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [-181.0, 181.0])
def test_longitude_outside_the_physical_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["site"]["longitude_E"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.longitude_E"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [-181.0, 181.0])
def test_meridian_radiance_outside_the_physical_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["site"]["meridian_radiance"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.meridian_radiance"):
        load_project(path)


@pytest.mark.parametrize("bad_value", [-361.0, 361.0])
def test_north_rotation_outside_the_physical_range_is_refused(write_example, tmp_path, bad_value):
    def mutate(d):
        d["site"]["north_rotation_deg"] = bad_value
        return d

    path = write_example(tmp_path, mutate=mutate)
    with pytest.raises(ProjectError, match="site.north_rotation_deg"):
        load_project(path)


# ---------------------------------------------------------------------------
# performance.trace_scope: "storey" carries an unproven-equivalence warning
# ---------------------------------------------------------------------------

def test_trace_scope_storey_is_a_warning(write_example, tmp_path):
    def mutate(d):
        d["performance"]["trace_scope"] = "storey"
        return d

    path = write_example(tmp_path, mutate=mutate)
    project = load_project(path)
    assert any("trace_scope" in w for w in project.warnings)


def test_trace_scope_all_produces_no_warning(example_project_dir):
    # The unmutated example already uses "all".
    project = load_project(example_project_dir / "project.yaml")
    assert not any("trace_scope" in w for w in project.warnings)


# ---------------------------------------------------------------------------
# site.meridian_radiance (west positive) against site.longitude_E (east positive)
# ---------------------------------------------------------------------------

def _site(longitude_E, meridian_radiance):
    def mutate(d):
        d["site"]["longitude_E"] = longitude_E
        d["site"]["meridian_radiance"] = meridian_radiance
        return d
    return mutate


def test_a_meridian_with_the_east_positive_sign_is_refused(write_example, tmp_path):
    # A site at 25 E in a UTC+2 zone: the Radiance meridian is -30, not +30.
    path = write_example(tmp_path, mutate=_site(25.0, 30.0))
    with pytest.raises(ProjectError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "site.meridian_radiance" in message
    assert "west-positive" in message
    assert "-15" in message


def test_a_consistent_meridian_passes(write_example, tmp_path):
    project = load_project(write_example(tmp_path, mutate=_site(25.0, -30.0)))
    assert not any("meridian_radiance" in w for w in project.warnings)


def test_a_far_but_same_sign_meridian_is_only_a_warning(write_example, tmp_path):
    # 40 degrees from the site's own longitude, but the same sign: odd
    # (a political time zone far from the site), not a convention mix.
    project = load_project(write_example(tmp_path, mutate=_site(-80.0, 120.0)))
    assert any("site.meridian_radiance" in w for w in project.warnings)


def test_a_near_zero_meridian_far_from_the_site_is_only_a_warning(write_example, tmp_path):
    # Meridian 0 at a site 40 W: no sign to compare, so a warning, not an error.
    project = load_project(write_example(tmp_path, mutate=_site(-40.0, 0.0)))
    assert any("site.meridian_radiance" in w for w in project.warnings)


# ---------------------------------------------------------------------------
# weather.file: a Radiance .wea, a full year, hour stamps on the half hour
# ---------------------------------------------------------------------------

def _hour_lines(stamp=lambda h: h + 0.5):
    import calendar
    return [
        f"{month} {day} {stamp(hour)} 0 0"
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(2023, month)[1] + 1)
        for hour in range(24)
    ]


def _load_with_weather(write_example, tmp_path, text, name="example.wea"):
    def mutate(d):
        d["weather"]["file"] = f"weather/{name}"
        return d
    path = write_example(tmp_path, mutate=mutate)
    (path.parent / "weather" / name).write_text(text, newline="\n")
    return load_project(path)


def test_the_example_weather_file_passes(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")
    assert not any("weather.file" in w for w in project.warnings)


def test_an_epw_weather_file_is_refused(write_example, tmp_path):
    with pytest.raises(ProjectError) as excinfo:
        _load_with_weather(write_example, tmp_path, "LOCATION,Example\n", name="example.epw")
    message = str(excinfo.value)
    assert "weather.file" in message
    assert ".wea" in message and "epw2wea" in message


def test_a_short_weather_file_is_refused(write_example, tmp_path):
    from conftest import wea_text
    with pytest.raises(ProjectError, match=r"weather\.file.*8759 data line"):
        _load_with_weather(write_example, tmp_path, wea_text(hours=_hour_lines()[:-1]))


def test_a_weather_file_not_starting_on_january_1st_is_refused(write_example, tmp_path):
    from conftest import wea_text
    lines = _hour_lines()
    lines = lines[24:] + lines[:24]
    with pytest.raises(ProjectError, match=r"weather\.file.*January 1"):
        _load_with_weather(write_example, tmp_path, wea_text(hours=lines))


def test_a_weather_file_out_of_order_is_refused(write_example, tmp_path):
    from conftest import wea_text
    lines = _hour_lines()
    lines[100], lines[101] = lines[101], lines[100]
    with pytest.raises(ProjectError, match=r"weather\.file.*chronological"):
        _load_with_weather(write_example, tmp_path, wea_text(hours=lines))


def test_hour_ending_integer_stamps_are_refused(write_example, tmp_path):
    from conftest import wea_text
    lines = _hour_lines(stamp=lambda h: float(h + 1))
    with pytest.raises(ProjectError) as excinfo:
        _load_with_weather(write_example, tmp_path, wea_text(hours=lines))
    message = str(excinfo.value)
    assert "half hour" in message and "30 minutes" in message


def test_a_weather_file_without_its_header_is_refused(write_example, tmp_path):
    text = "\n".join(_hour_lines()) + "\n"
    with pytest.raises(ProjectError, match=r"weather\.file.*header.*latitude"):
        _load_with_weather(write_example, tmp_path, text)


def test_a_weather_file_for_another_site_is_a_warning(write_example, tmp_path):
    from conftest import wea_text
    # The example project is at 42 N, 80 W; this file is for 45 N, 80 E.
    project = _load_with_weather(
        write_example, tmp_path, wea_text(latitude=45.0, longitude_W=-80.0)
    )
    warning = next(w for w in project.warnings if "weather.file" in w)
    assert "latitude" in warning and "longitude" in warning
    assert "west-positive" in warning


def test_a_weather_file_with_east_positive_longitude_is_a_warning(write_example, tmp_path):
    from conftest import wea_text
    project = _load_with_weather(write_example, tmp_path, wea_text(longitude_W=-80.0))
    assert any("weather.file" in w and "longitude" in w for w in project.warnings)


def test_weather_checks_never_raise_on_an_unreadable_file(write_example, tmp_path):
    # Bytes that are not text at all still come back as an error, not a crash.
    def mutate(d):
        return d
    path = write_example(tmp_path, mutate=mutate)
    (path.parent / "weather" / "example.wea").write_bytes(b"\xff\xfe\x00garbage\x00")
    with pytest.raises(ProjectError, match=r"weather\.file"):
        load_project(path)
