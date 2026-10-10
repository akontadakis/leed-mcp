# tests/test_engine_fingerprint.py
"""Tests for input fingerprinting (layer 4): a stopped run resumes, a stale
run recomputes.

Every fixture here is a small text file with fictional, round-numbered
content -- never a real Radiance input. ``fingerprint_of`` hashes bytes,
never mtime, so most tests write a file, build a fingerprint, then check
``is_current`` against an ``output`` file and its ``<output>.fingerprint.json``
sidecar, written by ``write_sidecar``.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import pytest

from leed_daylight.constants import TOOL_VERSION
from leed_daylight.engine.fingerprint import (
    Fingerprint,
    FingerprintError,
    delete_sidecar,
    fingerprint_of,
    is_current,
    write_sidecar,
)


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# --- the seven required rows --------------------------------------------


def test_an_unchanged_input_is_current(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    assert is_current(output, fp) is True


def test_a_changed_input_is_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    _write(scene, "void polygon a\n0\n0\n9\nchanged\n")
    changed_fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})

    assert is_current(output, changed_fp) is False


def test_a_changed_parameter_is_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    other_fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 3})

    assert is_current(output, other_fp) is False


def test_a_changed_tool_version_is_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    newer_fp = dataclasses.replace(fp, tool_version="fictional-tool-9.9.9")

    assert is_current(output, newer_fp) is False


def test_a_missing_sidecar_is_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    # No write_sidecar call at all.

    assert is_current(output, fp) is False


def test_touching_a_file_does_not_make_it_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    # Rewrite the identical bytes; the mtime changes, the content does not
    # (simulates copying a project folder to a new location).
    identical_bytes = scene.read_bytes()
    scene.write_bytes(identical_bytes)

    touched_fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})

    assert is_current(output, touched_fp) is True


def test_a_missing_input_is_a_named_error(tmp_path):
    missing = tmp_path / "does_not_exist.rad"

    with pytest.raises(FileNotFoundError, match="scene"):
        fingerprint_of({"scene": missing}, {})


# --- controller rulings: sidecar shape, output missing, malformed sidecar --


def test_the_output_file_missing_is_stale_even_with_sidecar(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    output.unlink()

    assert is_current(output, fp) is False


def test_a_malformed_sidecar_is_stale_not_a_crash(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    sidecar = tmp_path / "result.dat.fingerprint.json"
    sidecar.write_text("{not valid json", encoding="utf-8", newline="\n")

    assert is_current(output, fp) is False


def test_a_sidecar_that_is_a_directory_is_stale_not_a_crash(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    sidecar = tmp_path / "result.dat.fingerprint.json"
    sidecar.mkdir()

    assert is_current(output, fp) is False


def test_a_different_set_of_input_labels_is_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    weather = _write(tmp_path / "weather.wea", "place fictional_site\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    other_fp = fingerprint_of(
        {"scene": scene, "weather": weather}, {"ambient_bounces": 5}
    )

    assert is_current(output, other_fp) is False


def test_the_sidecar_path_is_beside_the_output(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "nested" / "result.dat"
    output.parent.mkdir()
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)

    expected = tmp_path / "nested" / "result.dat.fingerprint.json"
    assert expected.exists()


def test_the_sidecar_is_json_with_sorted_keys(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"zebra": 1, "ambient": 2})
    write_sidecar(output, fp)

    sidecar = tmp_path / "result.dat.fingerprint.json"
    text = sidecar.read_text(encoding="utf-8")
    # sort_keys=True: "ambient" (inside parameters) precedes "zebra".
    assert text.index('"ambient"') < text.index('"zebra"')
    payload = json.loads(text)
    assert payload["tool_version"] == TOOL_VERSION


# --- hostile-input pass: parameters must be JSON-representable ----------


def test_a_path_parameter_is_a_named_refusal(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")

    with pytest.raises(FingerprintError):
        fingerprint_of({"scene": scene}, {"weird": pathlib.Path(".")})


def test_a_nan_parameter_is_a_named_refusal(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")

    with pytest.raises(FingerprintError):
        fingerprint_of({"scene": scene}, {"weird": math.nan})


def test_an_infinite_parameter_is_a_named_refusal(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")

    with pytest.raises(FingerprintError):
        fingerprint_of({"scene": scene}, {"weird": math.inf})


def test_a_non_string_parameter_key_is_a_named_refusal(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")

    with pytest.raises(FingerprintError):
        fingerprint_of({"scene": scene}, {"nested": {1: "bad"}})


def test_a_numpy_array_parameter_is_a_named_refusal(tmp_path):
    numpy = pytest.importorskip("numpy")
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")

    with pytest.raises(FingerprintError):
        fingerprint_of({"scene": scene}, {"weird": numpy.array([1, 2, 3])})


def test_a_directory_input_is_a_named_refusal(tmp_path):
    directory = tmp_path / "a_directory"
    directory.mkdir()

    with pytest.raises(FingerprintError, match="scene"):
        fingerprint_of({"scene": directory}, {})


def test_tuple_parameters_round_trip_as_lists(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({"scene": scene}, {"flags": ("a", "b", "c")})
    assert fp.parameters["flags"] == ["a", "b", "c"]

    write_sidecar(output, fp)

    assert is_current(output, fp) is True


def test_empty_inputs_and_parameters_are_allowed(tmp_path):
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")

    fp = fingerprint_of({}, {})
    write_sidecar(output, fp)

    assert is_current(output, fp) is True


def test_fingerprint_is_a_frozen_dataclass(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})

    assert isinstance(fp, Fingerprint)
    with pytest.raises(Exception):
        fp.tool_version = "changed"  # frozen: attribute assignment refused


# --- delete_sidecar (final-fix M1) --------------------------------------


def test_delete_sidecar_removes_an_existing_sidecar(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")
    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)
    sidecar = tmp_path / "result.dat.fingerprint.json"
    assert sidecar.exists()

    delete_sidecar(output)

    assert not sidecar.exists()


def test_delete_sidecar_on_a_missing_sidecar_is_not_an_error(tmp_path):
    output = tmp_path / "never_built.dat"

    delete_sidecar(output)  # must not raise


def test_a_deleted_sidecar_makes_the_output_stale(tmp_path):
    scene = _write(tmp_path / "scene.rad", "void polygon a\n0\n0\n9\n")
    output = tmp_path / "result.dat"
    output.write_text("fake result", newline="\n")
    fp = fingerprint_of({"scene": scene}, {"ambient_bounces": 5})
    write_sidecar(output, fp)
    assert is_current(output, fp) is True

    delete_sidecar(output)

    assert is_current(output, fp) is False
