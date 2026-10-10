# tests/test_golden_figures.py
"""The Word-report figures of the invented project are byte-identical to master (atlas spec 8.2, 15).

The goldens under tests/fixtures/golden_figures/ were written once by
tests/golden_figures.py on master, before the new display-list items. Every
figure of the six Word reports is built again here, on the code under test,
and painted with matplotlib, Pillow and the pure SVG writer: every output
must match its recorded sha256, and the kept figures their golden bytes.

Every PNG must also decode to its recorded pixels (``pixels_sha256.json``)
on every system. Its bytes are checked on macOS and Windows only: Pillow's
Linux wheels compress the same pixels into other PNG bytes (a different
zlib build), so on Linux the pixels are the check. The SVGs are text and
are checked byte for byte everywhere.
"""
from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
pytest.importorskip("matplotlib", reason="the report extras are not installed: pip install -e '.[report]'")
pytest.importorskip("PIL", reason="the report extras are not installed: pip install -e '.[report]'")

from golden_figures import (
    DIGESTS, GOLDEN_DIR, PIXEL_DIGESTS, decode_png, digests, kept_names, pixel_digests, render_figures,
)

#: Why PNG bytes are not compared on Linux (see the module docstring).
LINUX_PNG_BYTES = ("Linux: Pillow's Linux wheels compress the same pixels into other PNG bytes "
                   "(a different zlib); the pixels are compared instead")
ON_LINUX = sys.platform.startswith("linux")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    return render_figures(tmp_path_factory.mktemp("golden_figures"))


def _golden_files() -> set[str]:
    return {
        p.relative_to(GOLDEN_DIR).as_posix()
        for p in GOLDEN_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".")  # never a Finder .DS_Store
    }


def test_the_golden_folder_holds_the_kept_figures_and_the_digest_list():
    assert _golden_files() == {*kept_names(), DIGESTS, PIXEL_DIGESTS}


def test_the_digest_list_names_every_figure_of_the_six_reports(rendered):
    recorded = json.loads((GOLDEN_DIR / DIGESTS).read_text())
    assert sorted(recorded) == sorted(rendered)


def test_the_pixel_digest_list_names_every_png(rendered):
    recorded = json.loads((GOLDEN_DIR / PIXEL_DIGESTS).read_text())
    assert sorted(recorded) == sorted(name for name in rendered if name.endswith(".png"))


def _changed(recorded: dict[str, str], now: dict[str, str]) -> list[str]:
    return [name for name in recorded if now.get(name) != recorded[name]]


def test_every_svg_matches_its_recorded_digest(rendered):
    recorded = json.loads((GOLDEN_DIR / DIGESTS).read_text())
    svgs = {name: digest for name, digest in recorded.items() if name.endswith(".svg")}
    assert svgs
    assert _changed(svgs, digests(rendered)) == [], "these figures differ from master"


@pytest.mark.skipif(ON_LINUX, reason=LINUX_PNG_BYTES)
def test_every_png_matches_its_recorded_digest(rendered):
    recorded = json.loads((GOLDEN_DIR / DIGESTS).read_text())
    pngs = {name: digest for name, digest in recorded.items() if name.endswith(".png")}
    assert pngs
    assert _changed(pngs, digests(rendered)) == [], "these figures differ from master"


def test_every_png_has_its_recorded_pixels(rendered):
    recorded = json.loads((GOLDEN_DIR / PIXEL_DIGESTS).read_text())
    assert _changed(recorded, pixel_digests(rendered)) == [], "these figures' pixels differ from master"


@pytest.mark.parametrize("name", kept_names())
def test_a_kept_figure_is_byte_identical_to_its_golden(rendered, name):
    if ON_LINUX and name.endswith(".png"):
        pytest.skip(LINUX_PNG_BYTES)
    assert rendered[name] == (GOLDEN_DIR / name).read_bytes()


@pytest.mark.parametrize("name", [name for name in kept_names() if name.endswith(".png")])
def test_a_kept_png_has_the_pixels_of_its_golden(rendered, name):
    mine, golden = decode_png(rendered[name]), decode_png((GOLDEN_DIR / name).read_bytes())
    assert (mine.mode, mine.size) == (golden.mode, golden.size)
    a, b = mine.tobytes(), golden.tobytes()
    if a != b:
        differing = sum(x != y for x, y in zip(a, b))
        largest = max(abs(x - y) for x, y in zip(a, b))
        pytest.fail(f"{name}: {differing} of {len(a)} pixel bytes differ, by at most {largest}")
