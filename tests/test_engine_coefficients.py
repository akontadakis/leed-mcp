# tests/test_engine_coefficients.py
"""Tests for leed_daylight.engine.coefficients (T4.4): daylight-coefficient
matrices per window group, and the lux boundary.

A FakeRunner answers ``rtrace -version``, writes a placeholder octree for
every ``oconv`` call, and writes a canned Radiance-format ascii matrix for
every ``rfluxmtx``/``rcontrib`` call, sized from the sensor count on stdin
and (for ``rcontrib``) the modifier count in the ``-M`` file -- it never
touches a real Radiance binary.

The coefficient-flow tests (``coefficients``/``Option1Tracer``) build a
real ``PreparedModel`` over the fictional two-room building in
``tests/fixtures/mixed_glazing.rad`` (also used by ``test_build_prepare``),
loaded through the real ``load_project`` from the fictional
``tests/fixtures/project_example.yaml`` (via conftest's ``write_example``),
so the ambient/direct strings a test checks for are the real, loaded
values, not hand-typed ones. The scope-only tests (``sensors_in_scope``)
construct ``WindowGroup``/``RoomGrid`` directly, per the brief.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
from types import MappingProxyType

import numpy as np
import pytest

from leed_daylight.build.grid import RoomGrid
from leed_daylight.build.groups import WindowGroup
from leed_daylight.build.prepare import prepare
from leed_daylight.engine.coefficients import (
    CoefficientError,
    GroupMatrices,
    Option1Tracer,
    coefficients,
    read_matrix,
    sensors_in_scope,
    to_lux,
)
from leed_daylight.drivers.run_option1 import run_option1
from leed_daylight.engine.radiance import Radiance, RunResult
from leed_daylight.project.config import Performance, load_project
from leed_daylight.project.rules import load_rules
from leed_daylight.project.spaces import Space
from leed_daylight.scene.reader import read_scene
from leed_daylight.scene.storeys import detect_rooms

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
MIXED_GLAZING = FIXTURES / "mixed_glazing.rad"

ROLES = {
    "wall_paint": "wall",
    "ceiling_tile": "ceiling",
    "floor_carpet": "floor",
    "glass_clear": "glazing",
    "glass_partition": "glazing_interior",
}

#: Must match leed_daylight.engine.coefficients._BLACK_MATERIAL_NAME --
#: duplicated here (not imported) so this test exercises the collision
#: purely through the module's public surface.
_BLACK_MATERIAL_NAME = "leed_daylight_black"


# -- a fake Radiance runner --------------------------------------------------


def _ascii_matrix_bytes(nrows: int, ncols: int, value: float = 0.1) -> bytes:
    header = f"#?RADIANCE\nNROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT=ascii\n\n"
    if nrows == 0 or ncols == 0:
        return header.encode("ascii")
    row = " ".join([f"{value} {value} {value}"] * ncols)
    body = "\n".join([row] * nrows) + "\n"
    return (header + body).encode("ascii")


def _float_matrix_bytes(nrows, ncols, value, big_endian=False) -> bytes:
    order = "big_endian" if big_endian else "little_endian"
    header = (
        f"#?RADIANCE\nNROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT=float\n"
        f"BYTEORDER={order}\n\n"
    )
    dtype = np.dtype(">f4" if big_endian else "<f4")
    arr = np.full((nrows, ncols, 3), value, dtype=dtype)
    return header.encode("ascii") + arr.tobytes()


def _bigendian_flag_matrix_bytes(nrows, ncols, value, big_endian) -> bytes:
    """A float matrix with the header a real rcontrib/gendaymtx writes:
    ``BigEndian=0|1``, not ``BYTEORDER``."""
    header = (
        f"#?RADIANCE\nNROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT=float\n"
        f"BigEndian={int(big_endian)}\n\n"
    )
    arr = np.full((nrows, ncols, 3), value, dtype=np.dtype(">f4" if big_endian else "<f4"))
    return header.encode("ascii") + arr.tobytes()


def _double_matrix_bytes(nrows, ncols, value, big_endian=False) -> bytes:
    order = "big_endian" if big_endian else "little_endian"
    header = (
        f"#?RADIANCE\nNROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT=double\n"
        f"BYTEORDER={order}\n\n"
    )
    dtype = np.dtype(">f8" if big_endian else "<f8")
    arr = np.full((nrows, ncols, 3), value, dtype=dtype)
    return header.encode("ascii") + arr.tobytes()


class FakeRunner:
    """Records every call (and its stdin); answers ``-version``, ``oconv``,
    ``rfluxmtx`` and ``rcontrib`` without ever touching a real Radiance
    binary."""

    def __init__(self, *, version="fake_radiance 1.2.3", sky_patches=2, value=0.1):
        self.calls: list[list[str]] = []
        self.stdins: list[bytes | None] = []
        self.version = version
        self.sky_patches = sky_patches
        self.value = value

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        self.stdins.append(stdin)
        if "-version" in argv:
            return RunResult(0, self.version.encode("utf-8"), b"")
        if stdout_path is None:
            return RunResult(0, b"", b"")
        stdout_path = pathlib.Path(stdout_path)
        n_sensors = stdin.decode("ascii").count("\n") if stdin else 0
        if argv[0] == "oconv":
            stdout_path.write_bytes(b"fake octree bytes")
        elif argv[0] == "rfluxmtx":
            stdout_path.write_bytes(_ascii_matrix_bytes(n_sensors, self.sky_patches, self.value))
        elif argv[0] == "rcontrib":
            mod_path = pathlib.Path(argv[argv.index("-M") + 1])
            modifiers = [l for l in mod_path.read_text().splitlines() if l.strip()]
            stdout_path.write_bytes(_ascii_matrix_bytes(n_sensors, len(modifiers), self.value))
        else:
            stdout_path.write_bytes(b"")
        return RunResult(0, b"", b"")


def _rfluxmtx_calls(runner):
    return [c for c in runner.calls if c and c[0] == "rfluxmtx"]


def _rcontrib_calls(runner):
    return [c for c in runner.calls if c and c[0] == "rcontrib"]


def _oconv_calls(runner):
    return [c for c in runner.calls if c and c[0] == "oconv"]


# -- a fictional two-room prepared model, real files on disk -----------------


def _load_mixed_glazing():
    prims = read_scene([MIXED_GLAZING], "m")
    rooms = detect_rooms(prims, ROLES)
    return prims, rooms


def _make_spaces():
    return [
        Space(
            floor="L0", room="office_a", space_type="office", regularly_occupied=True,
            area_m2=24.0, centroid_x=3.0, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True}, reason="",
        ),
        Space(
            floor="L0", room="office_b", space_type="office", regularly_occupied=True,
            area_m2=16.0, centroid_x=8.2, centroid_y=2.0, level_z=0.0,
            view_preserving_shades=False, merge_into=None,
            include={"base_case": True}, reason="",
        ),
    ]


def _prepared_model(tmp_path, write_example, *, mutate=lambda d: d):
    """A real prepared model: the fictional Example Tower project
    (tests/fixtures/project_example.yaml) over the fictional two-room
    building (tests/fixtures/mixed_glazing.rad). Yields (prepared, project).
    """
    project_path = write_example(tmp_path, mutate)
    project = load_project(project_path)
    prims, rooms = _load_mixed_glazing()
    out_dir = tmp_path / "prepared"
    prepared = prepare(project, prims, ROLES, rooms, _make_spaces(), "base_case", out_dir)
    return prepared, project


def _write_suns_fixture(tmp_path, *, modifiers, hours, rgb, name="suns.mtx"):
    suns_path = tmp_path / name
    # sun_matrix's own content is unread by coefficients.py; only the .mod
    # and .suns.json sidecars are (R-P34 step 5, sky.read_suns).
    suns_path.write_text("void light dummy\n0\n0\n3 1 1 1\n", encoding="utf-8", newline="\n")
    (tmp_path / f"{name}.mod").write_text(
        "".join(f"{m}\n" for m in modifiers), encoding="utf-8",
    newline="\n")
    records = [
        {"modifier": m, "hour": h, "rgb": list(r)}
        for m, h, r in zip(modifiers, hours, rgb)
    ]
    (tmp_path / f"{name}.suns.json").write_text(json.dumps(records), encoding="utf-8", newline="\n")
    return suns_path


def _write_sky_matrix_fixture(path, *, nrows, ncols, value=0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ascii_matrix_bytes(nrows, ncols, value))
    return path


def _make_tracer(
    tmp_path, prepared, project, runner, *, n_hours=8,
    suns_modifiers=("solar600",), suns_hours=(3,),
    suns_rgb=((500.0, 500.0, 500.0),),
):
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=list(suns_modifiers), hours=list(suns_hours),
        rgb=[list(r) for r in suns_rgb],
    )
    sky_path = _write_sky_matrix_fixture(
        tmp_path / "sky.mtx", nrows=runner.sky_patches, ncols=n_hours,
    )
    return Option1Tracer(rad, project, prepared, sky=sky_path, suns=suns), rad, suns, sky_path


# -- hand-built WindowGroup/RoomGrid for the scope-only tests ----------------


def _grid(room: str, storey: int, n_points: int = 1) -> RoomGrid:
    pts = np.array([[float(i), 0.0, 0.0] for i in range(n_points)])
    return RoomGrid(
        room=room, points=pts, spacing_m=0.5, cell_area_m2=0.25,
        room_area_m2=10.0, storey=storey, level_z=float(storey) * 3.0,
    )


def _group(group_id: str, room: str, storey: int) -> WindowGroup:
    return WindowGroup(
        group_id=group_id, room=room, storey=storey, azimuth_bin_deg=180.0,
        openings=(), outward_normals=(), area_m2=1.0,
    )


# =============================================================================
# The brief's ten rows
# =============================================================================


def test_rgb_becomes_lux_at_this_boundary():
    matrix = np.ones((1, 1, 3))
    sky = np.ones((1, 1, 3))

    result = to_lux(matrix, sky)

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(178.9)  # 47.4 + 119.9 + 11.6


def test_the_result_is_points_by_hours():
    matrix = np.full((5, 3, 3), 0.2)
    sky = np.full((3, 8, 3), 0.5)

    result = to_lux(matrix, sky)

    assert result.shape == (5, 8)


def test_the_ambient_parameters_reach_the_command(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)

    argv = _rfluxmtx_calls(runner)[0]
    for token in shlex.split(project.radiance.ambient_full):
        assert token in argv, (token, argv)


def test_the_direct_run_uses_ab_zero(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)

    argv = _rcontrib_calls(runner)[0]
    assert "-ab" in argv
    assert argv[argv.index("-ab") + 1] == "0"


def test_a_direct_option_cannot_turn_the_sun_pass_into_an_inter_reflected_one(
    tmp_path, write_example,
):
    prepared, project = _prepared_model(tmp_path, write_example)
    project = dataclasses.replace(
        project, radiance=dataclasses.replace(project.radiance, direct="-ab 1 -ad 64"),
    )
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )

    coefficients(rad, prepared, prepared.groups[0], prepared.grids, project, False, suns=suns)

    argv = _rcontrib_calls(runner)[0]
    last_ab = len(argv) - 1 - argv[::-1].index("-ab")
    assert argv[last_ab + 1] == "0"


def _with_aux_primitive(prepared, cal_name):
    """Append a brightfunc naming ``cal_name`` to the prepared materials."""
    text = prepared.materials_file.read_text()
    prepared.materials_file.write_text(
        text + f"\nvoid brightfunc aux_fn\n2 f {cal_name}\n0\n0\n"
    , newline="\n")


def _coefficients_runner_calls(tmp_path, prepared, project, version="fake_radiance 1.2.3"):
    runner = FakeRunner(version=version)
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    coefficients(rad, prepared, prepared.groups[0], prepared.grids, project, False, suns=suns)
    return len(_rfluxmtx_calls(runner)) + len(_rcontrib_calls(runner))


def test_a_radiance_upgrade_rebuilds_the_matrices(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)

    assert _coefficients_runner_calls(tmp_path, prepared, project) == 2
    assert _coefficients_runner_calls(tmp_path, prepared, project) == 0
    assert _coefficients_runner_calls(tmp_path, prepared, project, "other_radiance 9.9") == 2


def test_a_changed_auxiliary_file_rebuilds_the_matrices(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    cal = pathlib.Path(project.scene.files[0]).parent / "aux.cal"
    cal.write_text("f(x) = 1;\n", newline="\n")
    _with_aux_primitive(prepared, "aux.cal")

    assert _coefficients_runner_calls(tmp_path, prepared, project) == 2
    assert _coefficients_runner_calls(tmp_path, prepared, project) == 0
    cal.write_text("f(x) = 2;\n", newline="\n")
    assert _coefficients_runner_calls(tmp_path, prepared, project) == 2


def test_a_missing_auxiliary_file_does_not_stop_the_fingerprint(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    _with_aux_primitive(prepared, "nowhere.cal")

    assert _coefficients_runner_calls(tmp_path, prepared, project) == 2
    assert _coefficients_runner_calls(tmp_path, prepared, project) == 0


def test_storey_scope_traces_only_its_own_storey():
    grids = [_grid("g0", storey=0), _grid("g1", storey=1), _grid("g2", storey=2)]
    group = _group("grp_180", "g1", storey=1)
    performance = Performance(trace_scope="storey", linked_zones=MappingProxyType({}))

    scope = sensors_in_scope(group, grids, performance)

    assert [g.room for g in scope] == ["g1"]


def test_a_linked_zone_is_added_to_storey_scope():
    grids = [_grid("Atrium", storey=0), _grid("g1", storey=1), _grid("g2", storey=2)]
    group = _group("grp_180", "g1", storey=1)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({1: ("Atrium",)}),
    )

    scope = sensors_in_scope(group, grids, performance)

    assert [g.room for g in scope] == ["Atrium", "g1"]


def test_all_scope_traces_every_storey():
    grids = [_grid("Atrium", storey=0), _grid("g1", storey=1), _grid("g2", storey=2)]
    group = _group("grp_180", "g1", storey=1)
    performance = Performance(
        trace_scope="all", linked_zones=MappingProxyType({1: ("Atrium",)}),
    )

    scope = sensors_in_scope(group, grids, performance)

    assert [g.room for g in scope] == ["Atrium", "g1", "g2"]


def test_the_sensor_order_is_recorded(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    assert len(prepared.grids) == 2  # office_a, office_b -- "two rooms"
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)

    expected = tuple(
        room
        for grid in prepared.grids
        for room in [grid.room] * int(grid.points.shape[0])
    )
    assert matrices.sensor_index == expected
    sensors_file = matrices.sky.parent / f"{matrices.sky.name}.sensors.txt"
    assert sensors_file.read_text().splitlines() == list(expected)


def test_a_group_whose_blinds_never_close_is_traced_once(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)
    open_schedule = np.ones(int(hours.sum()), dtype=bool)

    _, _, closed_lux = tracer.sky(group, hours, open_schedule)

    assert closed_lux is None
    assert len(_rfluxmtx_calls(runner)) == 1


def test_nothing_above_layer_4_receives_rgb(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(
        tmp_path, prepared, project, runner, n_hours=8,
        suns_modifiers=["solar600", "solar700"], suns_hours=[3, 4],
        suns_rgb=[(500.0, 500.0, 500.0), (300.0, 300.0, 300.0)],
    )
    group = prepared.groups[0]

    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    for field in dataclasses.fields(GroupMatrices):
        assert not isinstance(getattr(matrices, field.name), np.ndarray)

    lux = to_lux(np.ones((1, 1, 3)), np.ones((1, 1, 3)))
    assert lux.ndim == 2

    hours = np.zeros(8, dtype=bool)
    hours[3] = True
    hours[4] = True
    sensor_index, direct_lux = tracer.direct(group, hours)
    assert isinstance(sensor_index, tuple)
    assert direct_lux.ndim == 2

    open_schedule = np.array([True, False])
    _, open_lux, closed_lux = tracer.sky(group, hours, open_schedule)
    assert open_lux.ndim == 2
    assert closed_lux.ndim == 2


# =============================================================================
# sensors_in_scope refusals (R-P36)
# =============================================================================


def test_a_non_integer_linked_zone_key_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({"abc": ("g0",)}),
    )
    with pytest.raises(CoefficientError, match="abc"):
        sensors_in_scope(group, grids, performance)


def test_a_negative_linked_zone_key_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({-1: ("g0",)}),
    )
    with pytest.raises(CoefficientError, match="-1"):
        sensors_in_scope(group, grids, performance)


def test_a_bool_linked_zone_key_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({True: ("g0",)}),
    )
    with pytest.raises(CoefficientError, match="bool"):
        sensors_in_scope(group, grids, performance)


def test_a_digit_string_linked_zone_key_is_accepted():
    grids = [_grid("Atrium", storey=0), _grid("g1", storey=1)]
    group = _group("grp", "g1", storey=1)
    # The key names the GROUP's own storey (1, as a digit string), whose
    # linked zone lists a room on a DIFFERENT storey (Atrium, storey 0).
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({"1": ("Atrium",)}),
    )
    scope = sensors_in_scope(group, grids, performance)
    assert [g.room for g in scope] == ["Atrium", "g1"]


def test_a_linked_zone_storey_with_no_grid_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({5: ("g0",)}),
    )
    with pytest.raises(CoefficientError, match="storey 5"):
        sensors_in_scope(group, grids, performance)


def test_a_linked_zone_room_not_among_the_grids_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="storey", linked_zones=MappingProxyType({0: ("ghost_room",)}),
    )
    with pytest.raises(CoefficientError, match="ghost_room"):
        sensors_in_scope(group, grids, performance)


def test_an_unknown_trace_scope_is_refused():
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(trace_scope="nonsense", linked_zones=MappingProxyType({}))
    with pytest.raises(CoefficientError, match="trace_scope"):
        sensors_in_scope(group, grids, performance)


def test_linked_zones_are_validated_even_under_scope_all():
    # R-P36: "validates EVERY key and room on every call, whatever
    # trace_scope" -- an "all" scope must still catch a bad linked_zones
    # entry, even though it would never consult it.
    grids = [_grid("g0", storey=0)]
    group = _group("grp", "g0", storey=0)
    performance = Performance(
        trace_scope="all", linked_zones=MappingProxyType({0: ("ghost_room",)}),
    )
    with pytest.raises(CoefficientError, match="ghost_room"):
        sensors_in_scope(group, grids, performance)


# =============================================================================
# to_lux refusals
# =============================================================================


def test_to_lux_refuses_a_wrong_ndim_matrix():
    with pytest.raises(CoefficientError, match="matrix"):
        to_lux(np.ones((2, 3)), np.ones((3, 4, 3)))


def test_to_lux_refuses_a_wrong_ndim_sky():
    with pytest.raises(CoefficientError, match="sky"):
        to_lux(np.ones((2, 3, 3)), np.ones((3, 4)))


def test_to_lux_refuses_a_matrix_last_axis_not_3():
    with pytest.raises(CoefficientError, match="matrix"):
        to_lux(np.ones((2, 3, 4)), np.ones((3, 4, 3)))


def test_to_lux_refuses_a_sky_last_axis_not_3():
    with pytest.raises(CoefficientError, match="sky"):
        to_lux(np.ones((2, 3, 3)), np.ones((3, 4, 4)))


def test_to_lux_refuses_a_shape_mismatch_naming_both_shapes():
    with pytest.raises(CoefficientError, match=r"\(2, 3, 3\).*\(5, 4, 3\)"):
        to_lux(np.ones((2, 3, 3)), np.ones((5, 4, 3)))


def test_to_lux_refuses_nan_in_the_matrix():
    matrix = np.ones((1, 1, 3))
    matrix[0, 0, 0] = np.nan
    with pytest.raises(CoefficientError, match="matrix"):
        to_lux(matrix, np.ones((1, 1, 3)))


def test_to_lux_refuses_inf_in_the_sky():
    sky = np.ones((1, 1, 3))
    sky[0, 0, 0] = np.inf
    with pytest.raises(CoefficientError, match="sky"):
        to_lux(np.ones((1, 1, 3)), sky)


# =============================================================================
# read_matrix: the three formats, and refusals
# =============================================================================


def test_read_matrix_reads_ascii(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(_ascii_matrix_bytes(2, 3, value=0.5))

    result = read_matrix(path)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.float64
    assert result[0, 0, 0] == pytest.approx(0.5)


def test_read_matrix_reads_float(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(_float_matrix_bytes(2, 3, 0.25))

    result = read_matrix(path)

    assert result.shape == (2, 3, 3)
    assert result[1, 2, 0] == pytest.approx(0.25, rel=1e-6)


def test_read_matrix_reads_double(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(_double_matrix_bytes(2, 3, 0.75))

    result = read_matrix(path)

    assert result.shape == (2, 3, 3)
    assert result[1, 2, 0] == pytest.approx(0.75)


def test_read_matrix_honours_big_endian_byteorder(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(_double_matrix_bytes(1, 1, 3.5, big_endian=True))

    result = read_matrix(path)

    assert result[0, 0, 0] == pytest.approx(3.5)


def test_read_matrix_honours_the_bigendian_flag(tmp_path):
    big = tmp_path / "big.mtx"
    big.write_bytes(_bigendian_flag_matrix_bytes(1, 2, 2.5, big_endian=True))
    little = tmp_path / "little.mtx"
    little.write_bytes(_bigendian_flag_matrix_bytes(1, 2, 2.5, big_endian=False))

    assert read_matrix(big)[0, 1, 2] == pytest.approx(2.5)
    assert read_matrix(little)[0, 1, 2] == pytest.approx(2.5)


def test_read_matrix_refuses_a_missing_file(tmp_path):
    with pytest.raises(CoefficientError, match="could not be read"):
        read_matrix(tmp_path / "does_not_exist.mtx")


def test_read_matrix_refuses_a_missing_header_key(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(b"#?RADIANCE\nNROWS=1\nNCOMP=3\nFORMAT=ascii\n\n0.1 0.1 0.1\n")
    with pytest.raises(CoefficientError, match="NCOLS"):
        read_matrix(path)


def test_read_matrix_refuses_ncomp_not_3(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(
        b"#?RADIANCE\nNROWS=1\nNCOLS=1\nNCOMP=4\nFORMAT=ascii\n\n0.1 0.1 0.1 0.1\n"
    )
    with pytest.raises(CoefficientError, match="NCOMP"):
        read_matrix(path)


def test_read_matrix_refuses_a_data_size_mismatch(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(b"#?RADIANCE\nNROWS=2\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n0.1 0.1 0.1\n")
    with pytest.raises(CoefficientError, match="expected"):
        read_matrix(path)


def test_read_matrix_refuses_nan_in_the_data(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(b"#?RADIANCE\nNROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\nnan 0.1 0.1\n")
    with pytest.raises(CoefficientError, match="NaN"):
        read_matrix(path)


def test_read_matrix_refuses_an_unsupported_format(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(b"#?RADIANCE\nNROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=weird\n\nsomething\n")
    with pytest.raises(CoefficientError, match="FORMAT"):
        read_matrix(path)


def test_read_matrix_refuses_a_missing_blank_line(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(b"#?RADIANCE\nNROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n0.1 0.1 0.1\n")
    with pytest.raises(CoefficientError, match="header"):
        read_matrix(path)


# =============================================================================
# coefficients() refusals
# =============================================================================


def test_a_group_not_in_prepared_is_refused_by_name(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    ghost = _group("ghost_000", "office_a", storey=0)

    with pytest.raises(CoefficientError, match="ghost_000"):
        coefficients(rad, prepared, ghost, prepared.grids, project, False, suns=suns)


def test_an_empty_scope_is_refused(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="empty"):
        coefficients(rad, prepared, group, [], project, False, suns=suns)


def test_blinds_closed_with_no_blind_file_is_refused(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    prepared_no_blinds = dataclasses.replace(prepared, blind_files={})
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared_no_blinds.groups[0]

    with pytest.raises(CoefficientError, match=group.group_id):
        coefficients(
            rad, prepared_no_blinds, group, prepared_no_blinds.grids, project,
            True, suns=suns,
        )


def test_a_non_reinhart_sky_basis_is_refused(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    project = dataclasses.replace(project, sky=dataclasses.replace(project.sky, basis="TensorTree"))
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="Reinhart"):
        coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)


def test_a_black_material_name_collision_is_refused(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    with prepared.materials_file.open("a") as fh:
        fh.write(f"void plastic {_BLACK_MATERIAL_NAME}\n0\n0\n5 0.5 0.5 0.5 0.0 0.0\n")
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match=_BLACK_MATERIAL_NAME):
        coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    assert not _oconv_calls(runner)



#: The sky and ground receiver names the engine writes into
#: sky_receiver.rad; rfluxmtx(1) requires receiver modifiers to be unique
#: and not used elsewhere, so a scene name equal to one must be refused.
@pytest.mark.parametrize("name", [
    "leed_daylight_ground_glow", "leed_daylight_ground",
    "leed_daylight_sky_glow", "leed_daylight_sky",
])
def test_a_receiver_name_collision_is_refused(tmp_path, write_example, name):
    prepared, project = _prepared_model(tmp_path, write_example)
    with prepared.materials_file.open("a") as fh:
        fh.write(f"void plastic {name}\n0\n0\n5 0.5 0.5 0.5 0.0 0.0\n")
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match=name):
        coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    assert not _oconv_calls(runner)


def test_a_sun_modifier_name_collision_is_refused(tmp_path, write_example):
    # rcontrib collects every modifier listed in suns.mod; a scene material
    # with one of those names would add its own light to that sun's column.
    prepared, project = _prepared_model(tmp_path, write_example)
    with prepared.materials_file.open("a") as fh:
        fh.write("void plastic solar600\n0\n0\n5 0.5 0.5 0.5 0.0 0.0\n")
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="solar600"):
        coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    # The sky octree may already be built; the sun octree never is.
    assert not [argv for argv in _oconv_calls(runner) if str(suns) in argv]

# =============================================================================
# Option1Tracer: n_hours, hostile inputs, and the closed-blind trace
# =============================================================================


def test_option1_tracer_n_hours_reads_the_header_only(tmp_path):
    path = tmp_path / "sky.mtx"
    path.write_bytes(
        b"#?RADIANCE\nNROWS=2\nNCOLS=8760\nNCOMP=3\nFORMAT=ascii\n\n"
        b"this is not valid ascii matrix data at all, on purpose"
    )
    tracer = Option1Tracer(rad=None, project=None, prepared=None, sky=path, suns=path)

    assert tracer.n_hours == 8760


def test_direct_refuses_a_wrong_length_hours_mask(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="8"):
        tracer.direct(group, np.ones(5, dtype=bool))


def test_sky_refuses_a_wrong_length_hours_mask(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="8"):
        tracer.sky(group, np.ones(5, dtype=bool), np.ones(5, dtype=bool))


def test_sky_refuses_a_wrong_length_open_schedule(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    with pytest.raises(CoefficientError, match="open_schedule"):
        tracer.sky(group, hours, np.ones(3, dtype=bool))


def test_direct_refuses_a_sun_hour_index_out_of_range(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(
        tmp_path, prepared, project, runner, n_hours=8,
        suns_modifiers=["solar_bad"], suns_hours=[999], suns_rgb=[(500.0, 500.0, 500.0)],
    )
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    with pytest.raises(CoefficientError, match="solar_bad"):
        tracer.direct(group, hours)


def test_a_group_whose_blinds_do_close_is_traced_twice(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)
    open_schedule = np.array([True, True, False, True, True, True, True, True])

    _, open_lux, closed_lux = tracer.sky(group, hours, open_schedule)

    assert closed_lux is not None
    assert closed_lux.shape == open_lux.shape
    assert len(_rfluxmtx_calls(runner)) == 2


def test_a_group_with_empty_scope_is_never_traced(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    # trace_scope "storey" (not the fixture's default "all") so a group on
    # a storey with no grid genuinely has an empty scope.
    project = dataclasses.replace(
        project,
        performance=dataclasses.replace(
            project.performance, trace_scope="storey", linked_zones=MappingProxyType({}),
        ),
    )
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    # A hand-built group on a storey none of the prepared grids sit on.
    ghost_group = _group("ghost_099", "office_a", storey=99)
    hours = np.ones(8, dtype=bool)

    sensor_index, direct_lux = tracer.direct(ghost_group, hours)

    assert sensor_index == ()
    assert direct_lux.shape == (0, 8)
    assert runner.calls == []


def test_a_group_with_empty_scope_returns_none_for_closed(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    project = dataclasses.replace(
        project,
        performance=dataclasses.replace(
            project.performance, trace_scope="storey", linked_zones=MappingProxyType({}),
        ),
    )
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    ghost_group = _group("ghost_099", "office_a", storey=99)
    hours = np.ones(8, dtype=bool)
    open_schedule = np.zeros(8, dtype=bool)

    sensor_index, open_lux, closed_lux = tracer.sky(ghost_group, hours, open_schedule)

    assert sensor_index == ()
    assert open_lux.shape == (0, 8)
    assert closed_lux is None
    assert runner.calls == []


# =============================================================================
# hostile inputs: ragged / non-numeric arrays never leak a raw numpy error
# =============================================================================


def test_to_lux_refuses_a_ragged_matrix_by_name_not_a_raw_numpy_error():
    ragged = [[[1.0, 1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    with pytest.raises(CoefficientError, match="matrix"):
        to_lux(ragged, np.ones((2, 4, 3)))


def test_to_lux_refuses_a_non_numeric_sky_by_name_not_a_raw_numpy_error():
    non_numeric = [[["a", "b", "c"]]]
    with pytest.raises(CoefficientError, match="sky"):
        to_lux(np.ones((1, 1, 3)), non_numeric)


def test_direct_refuses_a_ragged_hours_mask_by_name(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]

    with pytest.raises(CoefficientError, match="hours"):
        tracer.direct(group, [[1, 2], [3]])


def test_sky_refuses_a_ragged_open_schedule_by_name(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    with pytest.raises(CoefficientError, match="open_schedule"):
        tracer.sky(group, hours, [[1, 2], [3]])


# =============================================================================
# final-fix C1 / M9: sensor stdin is real ASCII floats, and bad point data
# is a named refusal, never a raw unpacking error
# =============================================================================


def test_sensor_stdin_lines_parse_as_six_floats_matching_the_grid_points(
    tmp_path, write_example,
):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)

    rflux_index = next(i for i, c in enumerate(runner.calls) if c and c[0] == "rfluxmtx")
    stdin = runner.stdins[rflux_index]
    lines = stdin.decode("ascii").splitlines()
    all_points = np.concatenate([g.points for g in prepared.grids], axis=0)
    assert len(lines) == all_points.shape[0]
    for line, point in zip(lines, all_points):
        # Every token must parse as a plain float -- under numpy 2.x,
        # `f"{x!r}"` on a raw np.float64 writes "np.float64(1.5)", which
        # `float()` cannot parse at all (the exact defect this test would
        # have caught).
        tokens = line.split()
        assert len(tokens) == 6, line
        values = [float(t) for t in tokens]
        assert values[:3] == pytest.approx(point.tolist())
        assert values[3:] == [0.0, 0.0, 1.0]


def test_bad_point_data_is_a_named_coefficient_error(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]
    good_grid = next(g for g in prepared.grids if g.room == group.room)
    # A hand-built RoomGrid whose points are not shape (n, 3) -- only
    # reachable this way, per M9's own note (RoomGrid has no constructor
    # check).
    bad_grid = dataclasses.replace(good_grid, points=np.zeros((2, 2)))

    with pytest.raises(CoefficientError, match="shape"):
        coefficients(rad, prepared, group, [bad_grid], project, False, suns=suns)


# =============================================================================
# final-fix C2: realistic Radiance headers (a producing command line, and
# extra KEY=value lines) are parsed, not refused
# =============================================================================


def test_read_matrix_skips_a_command_line_header_row(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(
        b"#?RADIANCE\n"
        b"gendaymtx -m 4 -r 0.0 -of -c 1 1 1 site.wea\n"
        b"NROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n"
        b"0.1 0.1 0.1\n"
    )

    result = read_matrix(path)

    assert result.shape == (1, 1, 3)


def test_read_matrix_accepts_extra_key_value_header_lines(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(
        b"#?RADIANCE\n"
        b"rcontrib -I+ -ab 0 -faf -M mods.txt scene.oct\n"
        b"SOFTWARE= RADIANCE 5.4 fictional build\n"
        b"LATLONG= 50.0 -10.0\n"
        b"CAPDATE= 2025:01:01 00:00:00\n"
        b"NROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n"
        b"0.2 0.2 0.2\n"
    )

    result = read_matrix(path)

    assert result.shape == (1, 1, 3)
    assert result[0, 0, 0] == pytest.approx(0.2)


def test_read_matrix_still_refuses_a_missing_required_key_past_junk_lines(tmp_path):
    path = tmp_path / "m.mtx"
    path.write_bytes(
        b"#?RADIANCE\n"
        b"gendaymtx -m 4 -r 0.0 -of -c 1 1 1 site.wea\n"
        b"NROWS=1\nNCOMP=3\nFORMAT=ascii\n\n"  # NCOLS missing
        b"0.1 0.1 0.1\n"
    )

    with pytest.raises(CoefficientError, match="NCOLS"):
        read_matrix(path)


# =============================================================================
# final-fix I1: a "storey"-shaped scope and an "all"-shaped scope never
# share (or stale each other's) work folder
# =============================================================================


def test_different_scopes_use_different_paths_and_neither_goes_stale(
    tmp_path, write_example,
):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]
    grid_own = next(g for g in prepared.grids if g.room == group.room)
    grid_other = next(g for g in prepared.grids if g.room != group.room)

    narrow = coefficients(rad, prepared, group, [grid_own], project, False, suns=suns)
    wide = coefficients(
        rad, prepared, group, [grid_own, grid_other], project, False, suns=suns,
    )

    assert narrow.sky != wide.sky
    assert narrow.sun != wide.sun
    assert narrow.sky.exists()
    assert wide.sky.exists()

    # Running the wide scope must not have touched the narrow scope's own
    # files: re-running the narrow scope is still current (no new trace).
    calls_before = len(runner.calls)
    coefficients(rad, prepared, group, [grid_own], project, False, suns=suns)
    assert len(runner.calls) == calls_before


# =============================================================================
# final-fix M1: the old fingerprint sidecar is deleted BEFORE a rebuild
# =============================================================================


class _SidecarCheckingRunner(FakeRunner):
    """Like FakeRunner, but asserts the sky matrix's own sidecar is already
    gone the instant ``rfluxmtx`` is invoked -- proving deletion happens
    BEFORE the rebuild, not after (M1)."""

    def __init__(self, sidecar_path, **kwargs):
        super().__init__(**kwargs)
        self._sidecar_path = sidecar_path

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        if argv and argv[0] == "rfluxmtx":
            assert not self._sidecar_path.exists(), (
                "the old sidecar must be deleted before the rebuild starts"
            )
        return super().run(argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd)


def test_the_old_sidecar_is_deleted_before_coefficients_rebuilds(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    sidecar = matrices.sky.parent / f"{matrices.sky.name}.fingerprint.json"
    assert sidecar.exists()

    # Force a rebuild by changing a fingerprinted parameter (the ambient
    # string), and confirm the sidecar was already gone the instant the
    # rebuild's rfluxmtx call happened.
    project2 = dataclasses.replace(
        project,
        radiance=dataclasses.replace(project.radiance, ambient_full="-ab 9 -ad 99999"),
    )
    checking_runner = _SidecarCheckingRunner(sidecar)
    rad2 = Radiance(checking_runner, tmp_path / "run2.log")
    coefficients(rad2, prepared, group, prepared.grids, project2, False, suns=suns)

    assert sidecar.exists()  # the rebuild wrote a fresh one afterward


# =============================================================================
# final-fix M2: a malformed sun matrix (wrong NROWS/NCOLS) is a named
# refusal, never a raw numpy error
# =============================================================================


def test_direct_refuses_a_sun_matrix_row_count_mismatch(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    tracer.direct(group, hours)
    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    real_rows = read_matrix(matrices.sun).shape[0]
    matrices.sun.write_bytes(_ascii_matrix_bytes(real_rows + 1, 1, 0.1))

    with pytest.raises(CoefficientError, match="NROWS"):
        tracer.direct(group, hours)


def test_direct_refuses_a_sun_matrix_column_count_mismatch(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    tracer.direct(group, hours)
    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    real_matrix = read_matrix(matrices.sun)
    matrices.sun.write_bytes(_ascii_matrix_bytes(real_matrix.shape[0], 2, 0.1))

    with pytest.raises(CoefficientError, match="NCOLS"):
        tracer.direct(group, hours)


# =============================================================================
# final-fix I2: the tracer masks and caches the sky matrix ONCE per hours
# mask, and stores the full sky matrix as float32
# =============================================================================


def test_the_tracer_masks_the_sky_once_across_several_calls(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)
    open_schedule = np.ones(int(hours.sum()), dtype=bool)

    tracer.sky(group, hours, open_schedule)
    first_masked = tracer._sky_masked
    tracer.sky(group, hours, open_schedule)
    second_masked = tracer._sky_masked

    assert first_masked is not None
    assert second_masked is first_masked  # never re-sliced for the same mask


def test_the_sky_array_is_cached_as_float32_but_lux_stays_float64(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)
    open_schedule = np.ones(int(hours.sum()), dtype=bool)

    _, open_lux, _ = tracer.sky(group, hours, open_schedule)

    assert tracer._sky_array.dtype == np.float32
    assert open_lux.dtype == np.float64


# =============================================================================
# review item 6 (triage): run_option1(engine=None) end to end, over a fake
# runner that writes REALISTIC Radiance output -- the one path that
# exercises C1 (sensor stdin) and C2 (header command-line junk) through the
# REAL Option1Tracer, not a hand-fed fixture.
# =============================================================================


def _realistic_matrix_bytes(nrows, ncols, value, *, fmt="ascii", command):
    """A Radiance matrix with a REALISTIC header: a producing command line
    (no '='), then an ordinary KEY=value line, then the required keys --
    exactly the shape C2 fixes ``_parse_header_fields`` to accept."""
    header = (
        "#?RADIANCE\n"
        f"{command}\n"
        "SOFTWARE= fictional Radiance build\n"
        f"NROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT={fmt}\n\n"
    )
    if fmt == "ascii":
        if nrows == 0 or ncols == 0:
            return header.encode("ascii")
        row = " ".join([f"{value} {value} {value}"] * ncols)
        body = "\n".join([row] * nrows) + "\n"
        return (header + body).encode("ascii")
    dtype = np.dtype("<f4")
    arr = np.full((nrows, ncols, 3), value, dtype=dtype)
    return header.encode("ascii") + arr.tobytes()


#: One fictional sun, minute 0 -- matches the first (Jan-1, 00:00) data
#: line of the realistic .wea this test writes (final-fix I3).
_REALISTIC_SUN_TEXT = (
    "void light solar_realistic\n0\n0\n3 500 500 500\n\n"
    "solar_realistic source sun0\n0\n0\n4 0.1 0.2 0.9 0.5\n"
)


class _RealisticRunner:
    """A fake runner writing Radiance-REALISTIC output: headers carry a
    producing command line with no '=' (C2), and every sensor stdin is
    real ASCII floats a downstream rcontrib could actually parse (C1) --
    this drives ``run_option1(engine=None)``'s REAL ``Option1Tracer`` end
    to end, the one path no other fake in this module exercises."""

    def __init__(self, *, n_patches=2, n_hours=8760):
        self.calls: list[list[str]] = []
        self.stdins: list[bytes | None] = []
        self.n_patches = n_patches
        self.n_hours = n_hours

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        argv = list(argv)
        self.calls.append(argv)
        self.stdins.append(stdin)
        if "-version" in argv:
            return RunResult(0, b"fake_radiance 1.2.3 fictional build", b"")
        if argv[0] == "gendaymtx":
            if "-D" in argv:
                pathlib.Path(argv[argv.index("-D") + 1]).write_text(
                    _REALISTIC_SUN_TEXT, encoding="utf-8",
                newline="\n")
                pathlib.Path(argv[argv.index("-M") + 1]).write_text(
                    "solar_realistic\n", encoding="utf-8",
                newline="\n")
                return RunResult(0, b"", b"")
            pathlib.Path(stdout_path).write_bytes(
                _realistic_matrix_bytes(
                    self.n_patches, self.n_hours, 0.02, fmt="float",
                    command="gendaymtx -m 1 -r 0.0 -of -c 1 1 1 site.wea",
                )
            )
            return RunResult(0, b"", b"")
        if argv[0] == "oconv":
            pathlib.Path(stdout_path).write_bytes(b"fake octree bytes")
            return RunResult(0, b"", b"")
        if stdout_path is None:
            return RunResult(0, b"", b"")
        # Every stdin line MUST parse as exactly six ASCII floats (C1), or
        # this fake fails loudly here -- the same way a real rcontrib would
        # silently choke on `np.float64(1.5)` text.
        n_sensors = 0
        if stdin:
            for line in stdin.decode("ascii").splitlines():
                values = [float(t) for t in line.split()]
                assert len(values) == 6, line
            n_sensors = stdin.count(b"\n")
        if argv[0] == "rfluxmtx":
            pathlib.Path(stdout_path).write_bytes(
                _realistic_matrix_bytes(
                    n_sensors, self.n_patches, 0.05,
                    command="rfluxmtx -I+ -y 1 -faf - r.rad -i s.oct",
                )
            )
        elif argv[0] == "rcontrib":
            mod_path = pathlib.Path(argv[argv.index("-M") + 1])
            modifiers = [l for l in mod_path.read_text().splitlines() if l.strip()]
            pathlib.Path(stdout_path).write_bytes(
                _realistic_matrix_bytes(
                    n_sensors, len(modifiers), 300.0,
                    command="rcontrib -I+ -ab 0 -faf -M mods.txt s.oct",
                )
            )
        else:
            pathlib.Path(stdout_path).write_bytes(b"")
        return RunResult(0, b"", b"")


def test_run_option1_engine_none_end_to_end_over_realistic_radiance_output(
    tmp_path, write_example,
):
    prepared, project = _prepared_model(tmp_path, write_example)
    first_group = prepared.groups[0]
    prepared_one_room = dataclasses.replace(
        prepared,
        groups=(first_group,),
        grids=tuple(g for g in prepared.grids if g.room == first_group.room),
    )
    spaces = [s for s in _make_spaces() if s.room == first_group.room]

    # A real, tiny, fictional .wea: one Jan-1st 00:00 data line, matching
    # the fake gendaymtx -D sun named "sun0" (minute 0) -- final-fix I3.
    project.weather_file.parent.mkdir(parents=True, exist_ok=True)
    project.weather_file.write_text(
        "place fictional_site\nlatitude 50.0\nlongitude -10.0\n"
        "time_zone -15\nsite_elevation 0.0\nweather_data_file_units 1\n"
        "1 1 0.0 0 0\n",
        encoding="utf-8",
    newline="\n")
    runner = _RealisticRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    rules = load_rules(project.credit_rules_path)

    result = run_option1(
        rad, project, prepared_one_room, spaces, rules, "base_case", engine=None,
    )

    assert result.case == "base_case"
    # C1's stdin path and C2's header path were both actually exercised:
    rfluxmtx_calls = [c for c in runner.calls if c and c[0] == "rfluxmtx"]
    rcontrib_calls = [c for c in runner.calls if c and c[0] == "rcontrib"]
    assert rfluxmtx_calls
    assert rcontrib_calls


# =============================================================================
# --processes: -n N reaches rfluxmtx and rcontrib, results unchanged
# =============================================================================


def test_processes_reach_both_trace_commands(tmp_path, write_example, monkeypatch):
    monkeypatch.setattr("leed_daylight.engine.radiance.sys.platform", "linux")

    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )

    coefficients(
        rad, prepared, prepared.groups[0], prepared.grids, project, False,
        suns=suns, processes=4,
    )

    for argv in (_rfluxmtx_calls(runner)[0], _rcontrib_calls(runner)[0]):
        assert "-n" in argv
        assert argv[argv.index("-n") + 1] == "4"
    # rfluxmtx options must come before the '-' sender argument.
    fl = _rfluxmtx_calls(runner)[0]
    assert fl.index("-n") < fl.index("-")


def test_one_process_adds_no_n_option(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )

    coefficients(rad, prepared, prepared.groups[0], prepared.grids, project, False, suns=suns)

    assert "-n" not in _rfluxmtx_calls(runner)[0]
    assert "-n" not in _rcontrib_calls(runner)[0]


def test_the_process_count_does_not_invalidate_the_cache(tmp_path, write_example):
    """-n changes only how fast rcontrib runs, never its output, so a
    matrix built with one count is reused under another."""
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    group = prepared.groups[0]

    first = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    n_calls = len(runner.calls)
    second = coefficients(
        rad, prepared, group, prepared.grids, project, False, suns=suns, processes=8,
    )

    assert len(runner.calls) == n_calls
    assert np.array_equal(read_matrix(first.sky), read_matrix(second.sky))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_a_bad_process_count_is_refused(tmp_path, write_example, bad):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )
    with pytest.raises(CoefficientError, match="processes"):
        coefficients(
            rad, prepared, prepared.groups[0], prepared.grids, project, False,
            suns=suns, processes=bad,
        )
    with pytest.raises(CoefficientError, match="processes"):
        Option1Tracer(rad, project, prepared, sky=tmp_path / "sky.mtx", suns=suns, processes=bad)


# =============================================================================
# read_matrix: peak memory, identical values
# =============================================================================


def _random_float_matrix(tmp_path, nrows, ncols, *, fmt="float", big_endian=False, trailing=b""):
    rng = np.random.default_rng(7)
    values = rng.random((nrows, ncols, 3))
    code = ("f4" if fmt == "float" else "f8")
    dtype = np.dtype((">" if big_endian else "<") + code)
    order = "big_endian" if big_endian else "little_endian"
    header = (
        f"#?RADIANCE\nrcontrib -faf -n 4 scene.oct\nNROWS={nrows}\nNCOLS={ncols}\n"
        f"NCOMP=3\nFORMAT={fmt}\nBYTEORDER={order}\n\n"
    )
    raw = values.astype(dtype).tobytes()
    path = tmp_path / f"m_{fmt}_{big_endian}.mtx"
    path.write_bytes(header.encode("ascii") + raw + trailing)
    reference = np.frombuffer(raw, dtype=dtype).astype(np.float64).reshape(nrows, ncols, 3)
    return path, reference


@pytest.mark.parametrize("fmt", ["float", "double"])
@pytest.mark.parametrize("big_endian", [False, True])
def test_read_matrix_values_are_unchanged_across_chunks(tmp_path, monkeypatch, fmt, big_endian):
    import leed_daylight.engine.coefficients as coefficients_module

    # A tiny chunk forces many chunk boundaries, including a ragged last one.
    monkeypatch.setattr(coefficients_module, "_READ_CHUNK_VALUES", 7)
    path, reference = _random_float_matrix(tmp_path, 5, 11, fmt=fmt, big_endian=big_endian)

    result = read_matrix(path)

    assert result.dtype == np.float64
    assert np.array_equal(result, reference)


def test_read_matrix_accepts_one_trailing_newline(tmp_path):
    path, reference = _random_float_matrix(tmp_path, 3, 4, trailing=b"\n")
    assert np.array_equal(read_matrix(path), reference)


def test_read_matrix_float32_is_exact_for_float_files(tmp_path):
    path, reference = _random_float_matrix(tmp_path, 4, 6)
    result = read_matrix(path, dtype=np.float32)
    assert result.dtype == np.float32
    assert np.array_equal(result.astype(np.float64), reference)


def test_read_matrix_nan_is_refused_in_any_chunk(tmp_path, monkeypatch):
    import leed_daylight.engine.coefficients as coefficients_module

    monkeypatch.setattr(coefficients_module, "_READ_CHUNK_VALUES", 5)
    header = b"#?RADIANCE\nNROWS=2\nNCOLS=3\nNCOMP=3\nFORMAT=float\n\n"
    data = np.full(18, 0.5, dtype="<f4")
    data[16] = np.nan
    path = tmp_path / "nan.mtx"
    path.write_bytes(header + data.tobytes())
    with pytest.raises(CoefficientError, match="NaN"):
        read_matrix(path)


def test_read_matrix_peak_memory_is_the_result_not_three_copies(tmp_path, monkeypatch):
    """The old reader held the raw bytes and a float64 copy together (about
    3x the file). Now the float64 result plus one bounded chunk is the
    peak, and a float32 read is about 1x. The chunk is shrunk here so a
    small test file still spans many chunks, as a real matrix does."""
    import tracemalloc

    import leed_daylight.engine.coefficients as coefficients_module

    monkeypatch.setattr(coefficients_module, "_READ_CHUNK_VALUES", 16384)

    nrows, ncols = 400, 250  # 300,000 values, 1.2 MB of float data
    path, _reference = _random_float_matrix(tmp_path, nrows, ncols)
    n_bytes = nrows * ncols * 3 * 4

    tracemalloc.start()
    try:
        result = read_matrix(path)
        _current, peak64 = tracemalloc.get_traced_memory()
        del result
        tracemalloc.reset_peak()
        result32 = read_matrix(path, dtype=np.float32)
        _current, peak32 = tracemalloc.get_traced_memory()
        del result32
    finally:
        tracemalloc.stop()

    assert peak64 < 2.6 * n_bytes, peak64 / n_bytes
    assert peak32 < 1.6 * n_bytes, peak32 / n_bytes


def test_to_lux_of_a_float32_matrix_equals_its_float64_copy():
    rng = np.random.default_rng(3)
    matrix32 = rng.random((6, 5, 3)).astype(np.float32)
    sky = rng.random((5, 4, 3))
    assert np.array_equal(to_lux(matrix32, sky), to_lux(matrix32.astype(np.float64), sky))


class _FloatRunner(FakeRunner):
    """Writes -faf (FORMAT=float) matrices of varied values, like the real
    rfluxmtx/rcontrib, instead of ascii."""

    def run(self, argv, *, stdin=None, stdout_path=None, cwd=None):
        result = super().run(argv, stdin=stdin, stdout_path=stdout_path, cwd=cwd)
        if stdout_path is not None and argv[0] in ("rfluxmtx", "rcontrib"):
            path = pathlib.Path(stdout_path)
            text = path.read_bytes().decode("ascii")
            nrows = int(text.split("NROWS=")[1].split()[0])
            ncols = int(text.split("NCOLS=")[1].split()[0])
            rng = np.random.default_rng(nrows * 1000 + ncols)
            data = rng.random((nrows, ncols, 3)).astype("<f4")
            header = (
                f"#?RADIANCE\nNROWS={nrows}\nNCOLS={ncols}\nNCOMP=3\nFORMAT=float\n"
                "BYTEORDER=little_endian\n\n"
            )
            path.write_bytes(header.encode("ascii") + data.tobytes())
        return result


def test_tracer_lux_from_a_float_matrix_is_unchanged(tmp_path, write_example):
    """The tracer reads -faf matrices as float32 to save memory; its lux
    must be bit-identical to the old float64 read."""
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = _FloatRunner()
    tracer, rad, suns, sky_path = _make_tracer(
        tmp_path, prepared, project, runner, n_hours=8,
        suns_modifiers=("solar1", "solar2"), suns_hours=(2, 5),
        suns_rgb=((500.0, 400.0, 300.0), (200.0, 250.0, 150.0)),
    )
    group = prepared.groups[0]
    hours = np.ones(8, dtype=bool)

    _index, direct_lux = tracer.direct(group, hours)
    _index, open_lux, _closed = tracer.sky(group, hours, np.ones(8, dtype=bool))

    matrices = coefficients(rad, prepared, group, prepared.grids, project, False, suns=suns)
    sun64 = read_matrix(matrices.sun)
    expected_direct = np.zeros_like(direct_lux)
    from leed_daylight import units
    for s, hour in enumerate((2, 5)):
        rgb = np.array([(500.0, 400.0, 300.0), (200.0, 250.0, 150.0)][s])
        expected_direct[:, hour] += units.rgb_to_lux(sun64[:, s, :] * rgb)
    assert np.array_equal(direct_lux, expected_direct)

    sky64 = read_matrix(sky_path).astype(np.float32).astype(np.float64)
    assert np.array_equal(open_lux, to_lux(read_matrix(matrices.sky), sky64))


# =============================================================================
# the residual pass: every group's openings blacked out, traced once per unit
# =============================================================================


def test_residual_units_cover_every_grid_once():
    from leed_daylight.engine.coefficients import residual_units

    grids = [_grid("a", storey=0), _grid("b", storey=1), _grid("c", storey=0)]
    storey = Performance(trace_scope="storey", linked_zones=MappingProxyType({}))
    everything = Performance(trace_scope="all", linked_zones=MappingProxyType({}))

    assert [[g.room for g in u] for u in residual_units(grids, storey)] == [["a", "c"], ["b"]]
    assert [[g.room for g in u] for u in residual_units(grids, everything)] == [["a", "b", "c"]]


def test_the_residual_id_can_never_be_a_real_group_id():
    from leed_daylight.build._names import slug
    from leed_daylight.engine.coefficients import RESIDUAL_ID

    assert slug(RESIDUAL_ID) != RESIDUAL_ID


def test_the_residual_blacks_out_every_group_and_adds_no_blinds(tmp_path, write_example):
    from leed_daylight.engine.coefficients import RESIDUAL_ID, residual_coefficients
    from leed_daylight.scene.reader import read_scene as _read

    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    rad = Radiance(runner, tmp_path / "run.log")
    suns = _write_suns_fixture(
        tmp_path, modifiers=["solar600"], hours=[3], rgb=[[500.0, 500.0, 500.0]],
    )

    matrices = residual_coefficients(rad, prepared, prepared.grids, project, suns=suns)

    assert matrices.group_id == RESIDUAL_ID
    assert RESIDUAL_ID in matrices.sky.parts
    isolated = _read([matrices.sky.parent / "isolated.rad"], "m")
    by_name = {p.name: p for p in isolated}
    for group in prepared.groups:
        for opening in group.openings:
            assert by_name[opening.name].modifier == _BLACK_MATERIAL_NAME
    # Interior glazing belongs to no group and stays as it is.
    assert by_name["window_int"].modifier == "glass_partition"
    blind_files = {str(p) for p in prepared.blind_files.values()}
    for argv in _oconv_calls(runner):
        assert not blind_files & set(argv)
    assert len(_rfluxmtx_calls(runner)) == 1
    assert len(_rcontrib_calls(runner)) == 1

    # Cached like a group: a second call traces nothing.
    n_calls = len(runner.calls)
    residual_coefficients(rad, prepared, prepared.grids, project, suns=suns)
    assert len(runner.calls) == n_calls


def test_the_tracer_residual_traces_once_per_unit_not_per_group(tmp_path, write_example):
    prepared, project = _prepared_model(tmp_path, write_example)
    runner = FakeRunner()
    tracer, rad, suns, sky_path = _make_tracer(tmp_path, prepared, project, runner, n_hours=8)
    hours = np.ones(8, dtype=bool)

    units = list(tracer.residual(hours))

    assert len(units) == 1  # the fixture project's trace_scope is "all"
    (sensor_index, sky_lux, direct_lux), = units
    expected_index = tuple(
        g.room for g in prepared.grids for _ in range(int(g.points.shape[0]))
    )
    assert sensor_index == expected_index
    assert sky_lux.shape == (len(expected_index), 8)
    assert direct_lux.shape == (len(expected_index), 8)
    assert np.all(sky_lux > 0.0)
    # One residual rfluxmtx and one rcontrib, whatever the group count.
    assert len(_rfluxmtx_calls(runner)) == 1
    assert len(_rcontrib_calls(runner)) == 1


def test_the_isolated_scene_writer_quotes_awkward_words(tmp_path):
    from leed_daylight.engine.coefficients import _primitive_text
    from leed_daylight.scene.reader import read_scene as _read

    path = tmp_path / "isolated.rad"
    path.write_text(_primitive_text("void", "brightfunc", "dim", ("a b.cal", "x#y", "it's"), (), ()), newline="\n")
    [prim] = _read([path], "m")
    assert prim.sargs == ("a b.cal", "x#y", "it's")
