# tests/test_docs_acceptance.py
"""The guide and the README describe what the acceptance check added: the
sensor points file, the grid-area blind trigger, frosted interior glass,
and the two-run acceptance fixture."""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUNNING_A_PROJECT.md"
README = ROOT / "docs" / "PIPELINE.md"


@pytest.mark.parametrize("phrase", [
    "`points_file`",
    "**Sensor points from a file.**",
    "`space,x,y,z`",
    "floor level plus `work_plane_m`",
    "`trigger_area`",
    "point count times `spacing_m` squared",
    "**Frosted interior glass**",
    "`grid.points_file line",
    "`LEED_ACCEPTANCE_RUN`",
])
def test_the_guide_explains_the_acceptance_additions(phrase):
    assert phrase in GUIDE.read_text(encoding="utf-8")


def test_the_guide_no_longer_says_trans_needs_the_diffusing_role_alone():
    text = GUIDE.read_text(encoding="utf-8")
    assert "Diffusing glass must be a `trans` with the role `glazing_diffusing`, 7 real" not in text


def test_the_readme_names_the_run_variable():
    text = README.read_text(encoding="utf-8")
    assert "LEED_ACCEPTANCE_RUN=A" in text
    assert "tests/acceptance/test_fixture_option1.py" in text
