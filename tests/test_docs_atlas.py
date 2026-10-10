# tests/test_docs_atlas.py
"""The guide and the README describe the floor atlas and the Blender file
(atlas spec 16)."""
from __future__ import annotations

import pathlib
import textwrap

import pytest
import yaml

from leed_daylight.project.config import DrawingSpec, _load_atlas

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUNNING_A_PROJECT.md"
README = ROOT / "docs" / "PIPELINE.md"


@pytest.mark.parametrize("doc", [GUIDE, README], ids=lambda p: p.name)
@pytest.mark.parametrize("phrase", [
    "leed-daylight atlas ",
    "--no-blender",
    "`BLENDER`",
    "`/Applications/Blender.app/Contents/MacOS/Blender`",
    "Blender 4.2 or newer",
    "`atlas:`",
    "`atlas/`",
    "never `blender.log`",
    "`report --option views`",
])
def test_the_docs_explain_the_atlas(doc, phrase):
    assert phrase in doc.read_text(encoding="utf-8")


def test_the_guide_names_the_one_time_package_refusal():
    text = " ".join(GUIDE.read_text(encoding="utf-8").split())
    assert "**Reports made before the floor atlas was added have no atlas**" in text
    assert "asks for `report` once more" in text


def test_the_atlas_is_no_longer_listed_as_not_built():
    readme = README.read_text(encoding="utf-8")
    (line,) = [line for line in readme.splitlines() if line.startswith("Not built yet:")]
    assert "atlas" not in line and "Blender" not in line
    guide = GUIDE.read_text(encoding="utf-8")
    assert "are not built yet, nor is one command line" not in guide


def _guide_example() -> str:
    text = GUIDE.read_text(encoding="utf-8")
    (block,) = [b.split("```", 1)[0] for b in text.split("```yaml\n")[1:] if b.startswith("atlas:")]
    return block


def _readme_example() -> str:
    lines = README.read_text(encoding="utf-8").splitlines()
    start = lines.index("    atlas:")
    block = []
    for line in lines[start:]:
        if not line.startswith("    "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block))


@pytest.mark.parametrize("example", [_guide_example, _readme_example], ids=["guide", "readme"])
def test_the_drawing_example_is_a_valid_atlas_section(example):
    atlas = _load_atlas(yaml.safe_load(example()))

    assert atlas.drawings == (DrawingSpec(
        storey="L1", image="drawings/level1.png",
        pixel=((400.0, 1800.0), (3900.0, 1800.0)), model=((0.0, 0.0), (35.0, 0.0)), opacity=0.35,
    ),)
