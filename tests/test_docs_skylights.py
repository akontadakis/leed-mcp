# tests/test_docs_skylights.py
"""The guide and the README describe roof and diffusing glass (skylights
spec, sections 1, 4 and 5.2)."""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUNNING_A_PROJECT.md"
README = ROOT / "docs" / "PIPELINE.md"


def test_the_guide_no_longer_refuses_skylights():
    text = GUIDE.read_text(encoding="utf-8")
    assert "No skylights yet" not in text
    assert "Skylights are not modelled" not in text


@pytest.mark.parametrize("phrase", [
    "**Skylights, sloped glass and roof monitors.**",
    "`glazing_diffusing`",
    "`skylight_well`",
    "`skylight_well_reflectance`",
    "`{vertical: 0.95, sloped: 0.90, skylight: 0.85}`",
    "`<room>_sky`",
    "within 0.6 m of the room outline in plan",
    "**Projects made before skylights were supported.**",
    "the last `-ab`",
])
def test_the_guide_explains_roof_and_diffusing_glass(phrase):
    assert phrase in GUIDE.read_text(encoding="utf-8")


def test_the_readme_no_longer_lists_skylights_as_not_built():
    text = README.read_text(encoding="utf-8")
    (line,) = [line for line in text.splitlines() if line.startswith("Not built yet:")]
    assert "skylights" not in line
    assert "**Breaking change (roof and sloped glass):**" in text
