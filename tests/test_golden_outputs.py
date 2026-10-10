# tests/test_golden_outputs.py
"""With no splits.csv and no merge_into, command output and results are
byte-identical to master (merge-split spec 8, 9 and 10).

The goldens under tests/fixtures/golden_outputs/ were written once by
tests/golden_outputs.py on master, before the merge-split work: the printed
output of rooms, scan and prepare, the rooms template, the Option 1 and
Option 2 results, and the Quality Views results and plans of the invented
project of tests/deliver_folder.py. They are built again here, on the code
under test, and every file must match exactly on macOS, where they were
written. On Linux and Windows the C library rounds the last place of a
double differently (``same_up_to_last_digits``): there every byte but the
last digits of a decimal must match, and those to ``FLOAT_REL_TOL``.
"""
from __future__ import annotations

import sys

import pytest

from golden_outputs import GOLDEN_DIR, OUTPUTS, render_outputs, same_up_to_last_digits


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
    golden = (GOLDEN_DIR / name).read_bytes()
    if sys.platform == "darwin" or not same_up_to_last_digits(rendered[name], golden):
        assert rendered[name] == golden


@pytest.mark.parametrize("rendered_text, golden_text, same", [
    # source: the sun azimuths of option2_results.json on Windows and Linux CI against the macOS golden
    (b'{"a": 51.856548327297226}', b'{"a": 51.85654832729723}', True),
    # source: as above
    (b'{"a": -63.63363133793085}', b'{"a": -63.63363133793086}', True),
    # source: invented, a change in the fifth significant digit
    (b'{"a": 51.8565}', b'{"a": 51.8566}', False),
    (b'{"n": 54}', b'{"n": 55}', False),
    (b'"sha": "ce9a61"', b'"sha": "c9659d"', False),
    (b'{"a": 1.0}', b'{"b": 1.0}', False),
    (b'{"a": 1.0, "b": 2}', b'{"a": 1.0}', False),
])
def test_only_the_last_digits_of_a_decimal_may_differ(rendered_text, golden_text, same):
    assert same_up_to_last_digits(rendered_text, golden_text) is same
