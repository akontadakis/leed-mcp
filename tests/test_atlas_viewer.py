# tests/test_atlas_viewer.py
"""The ported 3D viewer (atlas spec 9, 10, 15).

The viewer was ported from earlier private work and stripped of every
project value: it holds no storey code list, no work-plane height, no grid
spacing, no hour count, no fixed scale ticks and no colour ramp; all of them
come from the page's payload. It reads only ``window.__ATLAS__`` and the
element ``#atlas-3d`` (the DOM contract the page of Task 6 provides), and it
makes no network request. A client name is caught by the private deny-list
scan (tests/test_confidentiality.py), which reads every tracked file,
these two included; it is never written into a test.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / "leed_daylight" / "deliver" / "atlas" / "web"
FILES = ("viewer.js", "viewer.css")


def _text(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", FILES)
def test_the_viewer_holds_no_project_value(name):
    text = _text(name)
    for value in ("0.762", "3,650", "3650"):
        assert value not in text
    assert not re.search(r"(?<![\d.])0\.6(?!\d)", text), "a fixed grid spacing"
    assert not re.search(r"""(["'])[A-Z0-9]{1,3}\1(\s*,\s*(["'])[A-Z0-9]{1,3}\3){2,}""", text), "a storey code list"
    assert not re.search(r"""["']\d{2,}["']""", text), "a fixed scale tick"
    assert "GROUND FLOOR" not in text.upper()


@pytest.mark.parametrize("name", FILES)
def test_the_viewer_makes_no_network_request(name):
    text = _text(name)
    assert "http:" not in text and "https:" not in text
    assert "url(" not in text
    assert "@import" not in text
    assert "@font-face" not in text


def test_the_colour_ramp_comes_from_the_payload():
    assert "linear-gradient" not in _text("viewer.css")
    assert not re.search(r"#[0-9a-fA-F]{6}", _text("viewer.js"))
    assert "g.ramp.map" in _text("viewer.js")


def test_the_viewer_reads_only_the_payload_and_its_element():
    js = _text("viewer.js")
    assert set(re.findall(r"getElementById\(\"([^\"]+)\"\)", js)) == {"atlas-3d"}
    assert "document.querySelector" not in js
    assert set(re.findall(r"window\.(__\w+__)", js)) == {"__ATLAS__"}
    for name in ("D.blob", "D.parts", "D.quant", "D.kinds", "D.storeys", "D.rooms", "D.layers", "D.cell", "D.work"):
        assert name in js


def test_the_viewer_says_so_without_webgl():
    assert "This browser has no WebGL, so the 3D view cannot be drawn" in _text("viewer.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_viewer_parses():
    result = subprocess.run(["node", "--check", str(WEB / "viewer.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
