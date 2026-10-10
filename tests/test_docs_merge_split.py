# tests/test_docs_merge_split.py
"""The guide and the README describe splitting and merging rooms
(merge-split spec 11)."""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUNNING_A_PROJECT.md"
README = ROOT / "docs" / "PIPELINE.md"


@pytest.mark.parametrize("phrase", [
    "**Split a room: `splits.csv`.**",
    "level_z,points,note",
    "**Metres, whatever `scene.units`**",
    "a split must end between panes",
    "each half at least 1 m2",
    "**Merge rooms: `merge_into`.**",
    "**space row**",
    "**piece row**",
    "`performance.linked_zones` names space rows only",
    "**Blinds respond to the whole space**",
    "`split lines applied: 1 (splits.csv)`",
    "`split lines: 1`",
    "`    pieces (merged): S0_R2 12.00 m2, S0_R3 8.00 m2`",
    "`config/splits.csv`",
    "dashed",
])
def test_the_guide_explains_splits_and_merges(phrase):
    assert phrase in GUIDE.read_text(encoding="utf-8")


@pytest.mark.parametrize("gone", [
    "Merging rooms is not supported yet",
    "merge_into is not supported for Quality Views",
    "`merge_into` must be empty for every ticked room",
])
def test_the_guide_no_longer_refuses_merge_into(gone):
    assert gone not in GUIDE.read_text(encoding="utf-8")


@pytest.mark.parametrize("phrase", ["`splits.csv`", "`merge_into`", "`config/splits.csv`", "`LEED_RADIANCE_SMOKE=1`"])
def test_the_readme_names_splits_and_merges(phrase):
    assert phrase in README.read_text(encoding="utf-8")


def test_the_readme_lists_split_and_merge_among_the_features():
    lines = README.read_text(encoding="utf-8").split("## Install", 1)[0].splitlines()
    assert any("split" in line and "merge" in line for line in lines)
