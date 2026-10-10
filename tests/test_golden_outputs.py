# tests/test_golden_outputs.py
"""With no splits.csv and no merge_into, command output and results are
byte-identical to master (merge-split spec 8, 9 and 10).

The goldens under tests/fixtures/golden_outputs/ were written once by
tests/golden_outputs.py on master, before the merge-split work: the printed
output of rooms, scan and prepare, the rooms template, the Option 1 and
Option 2 results, and the Quality Views results and plans of the invented
project of tests/deliver_folder.py. They are built again here, on the code
under test, and every file must match exactly.
"""
from __future__ import annotations

import pytest

from golden_outputs import GOLDEN_DIR, OUTPUTS, render_outputs


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    return render_outputs(tmp_path_factory.mktemp("golden_outputs"))


def test_there_is_one_golden_file_per_output_and_no_other():
    files = {
        p.relative_to(GOLDEN_DIR).as_posix()
        for p in GOLDEN_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".")  # never a Finder .DS_Store
    }
    assert files == set(OUTPUTS)


@pytest.mark.parametrize("name", OUTPUTS)
def test_outputs_are_byte_identical_to_master(rendered, name):
    assert rendered[name] == (GOLDEN_DIR / name).read_bytes()
