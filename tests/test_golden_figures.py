# tests/test_golden_figures.py
"""The Word-report figures of the invented project are byte-identical to master (atlas spec 8.2, 15).

The goldens under tests/fixtures/golden_figures/ were written once by
tests/golden_figures.py on master, before the new display-list items. Every
figure of the six Word reports is built again here, on the code under test,
and painted with matplotlib, Pillow and the pure SVG writer: every output
must match its recorded sha256, and the kept figures their golden bytes.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
pytest.importorskip("matplotlib", reason="the report extras are not installed: pip install -e '.[report]'")
pytest.importorskip("PIL", reason="the report extras are not installed: pip install -e '.[report]'")

from golden_figures import DIGESTS, GOLDEN_DIR, digests, kept_names, render_figures


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
    assert _golden_files() == {*kept_names(), DIGESTS}


def test_the_digest_list_names_every_figure_of_the_six_reports(rendered):
    recorded = json.loads((GOLDEN_DIR / DIGESTS).read_text())
    assert sorted(recorded) == sorted(rendered)


def test_every_figure_matches_its_recorded_digest(rendered):
    recorded = json.loads((GOLDEN_DIR / DIGESTS).read_text())
    now = digests(rendered)
    changed = [name for name in recorded if now.get(name) != recorded[name]]
    assert changed == [], f"these figures differ from master: {changed}"


@pytest.mark.parametrize("name", kept_names())
def test_a_kept_figure_is_byte_identical_to_its_golden(rendered, name):
    assert rendered[name] == (GOLDEN_DIR / name).read_bytes()
