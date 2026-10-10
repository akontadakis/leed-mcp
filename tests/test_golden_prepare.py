# tests/test_golden_prepare.py
"""Every scene master accepts prepares byte for byte as it did (skylights spec 5.12).

The goldens under tests/fixtures/golden_prepare/ were written once by
tests/golden_prepare.py on the commit before roof glass was supported.
Each invented all-vertical building is prepared again here, on the code
under test, and all five outputs (scene.rad, materials.rad, blinds/,
grids/, manifest.json) must match those files exactly: any difference
would change a manifest hash and silently invalidate every cached
daylight-coefficient matrix.
"""
from __future__ import annotations

import pytest

from golden_prepare import FIXTURE_NAMES, GOLDEN_DIR, TOP_LEVEL, prepare_golden, snapshot


def _visible(folder):
    return {p.name for p in folder.iterdir() if not p.name.startswith(".")}


def test_there_is_one_golden_folder_per_fixture_and_no_other():
    assert _visible(GOLDEN_DIR) == set(FIXTURE_NAMES)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_each_golden_folder_holds_the_five_outputs(name):
    assert _visible(GOLDEN_DIR / name) == TOP_LEVEL


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_prepare_is_byte_identical_to_the_golden(tmp_path, name):
    out_dir = tmp_path / name
    prepare_golden(name, out_dir)
    got = snapshot(out_dir)
    want = snapshot(GOLDEN_DIR / name)
    assert sorted(got) == sorted(want)
    changed = [path for path in want if got[path] != want[path]]
    assert changed == [], f"{name}: these prepared files differ from the golden: {changed}"
