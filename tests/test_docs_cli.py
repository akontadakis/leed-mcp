# tests/test_docs_cli.py
"""The README and the guide show the one command (CLI spec 9)."""
from __future__ import annotations

import pathlib

import pytest

from leed_daylight.cli import VERBS

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "PIPELINE.md", ROOT / "docs" / "RUNNING_A_PROJECT.md"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
@pytest.mark.parametrize("verb", [verb for verb, _ in VERBS])
def test_every_verb_is_shown_with_the_one_command(doc, verb):
    assert f"leed-daylight {verb} " in doc.read_text(encoding="utf-8")


def test_the_readme_names_the_check_script():
    assert ".venv/bin/python tools/check.py" in (ROOT / "docs" / "PIPELINE.md").read_text(encoding="utf-8")


def test_the_readme_says_how_to_install_the_command():
    assert 'uv sync --extra report --no-editable --reinstall-package leed-mcp' in (ROOT / "docs" / "PIPELINE.md").read_text(encoding="utf-8")


def test_the_readme_names_the_older_form_correctly():
    text = (ROOT / "docs" / "PIPELINE.md").read_text(encoding="utf-8")
    assert "The older form `.venv/bin/python -m leed_daylight.drivers.<verb>` still" in text
