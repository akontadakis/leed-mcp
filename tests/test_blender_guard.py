# tests/test_blender_guard.py
"""No test starts a real Blender by accident (atlas spec 15, Review Focus 3).

``tests/conftest.py`` sets ``BLENDER`` to ``NO_BLENDER``, a path that does
not exist, for every test (autouse). The Blender finder
(``engine/blender.py``) takes ``BLENDER`` whenever it is set and refuses a
path that is not there; it never goes on to ``PATH`` or /Applications. So a
Blender installed on the machine that runs the tests is never found, and
``report`` and ``package`` tests run the "Blender not found" path. Only the
opt-in smoke test (marked ``blender_smoke``), and only with
``LEED_BLENDER_SMOKE=1``, leaves ``BLENDER`` alone: setting the variable
never lifts the guard for any other test (final review M2).
"""
from __future__ import annotations

import os
import pathlib

import test_atlas_blender_smoke
from conftest import BLENDER_SMOKE_ENV, BLENDER_SMOKE_MARK, NO_BLENDER, blender_for_tests


def test_no_test_can_find_a_real_blender():
    # This test asks for no fixture: the guard is autouse, and it wins over
    # whatever BLENDER the shell that started pytest had set, also when
    # that shell set LEED_BLENDER_SMOKE=1.
    value = os.environ.get("BLENDER")
    assert value == str(NO_BLENDER)
    assert pathlib.Path(value).is_absolute()
    assert not pathlib.Path(value).exists()
    assert not pathlib.Path(value).parent.exists()


def test_the_guard_steps_aside_only_for_the_opt_in_smoke_test():
    for smoke_test in (False, True):
        assert blender_for_tests({}, smoke_test=smoke_test) == str(NO_BLENDER)
        assert blender_for_tests({BLENDER_SMOKE_ENV: "0"}, smoke_test=smoke_test) == str(NO_BLENDER)
        assert blender_for_tests({BLENDER_SMOKE_ENV: "yes"}, smoke_test=smoke_test) == str(NO_BLENDER)
        assert blender_for_tests({"BLENDER": "/elsewhere/blender"}, smoke_test=smoke_test) == str(NO_BLENDER)
    assert blender_for_tests({BLENDER_SMOKE_ENV: "1"}, smoke_test=False) == str(NO_BLENDER)
    assert blender_for_tests({BLENDER_SMOKE_ENV: "1"}, smoke_test=True) is None


def test_only_the_smoke_module_carries_the_smoke_mark():
    marks = test_atlas_blender_smoke.pytestmark
    assert BLENDER_SMOKE_MARK in [mark.name for mark in marks]
    here = pathlib.Path(__file__).parent
    carriers = sorted(p.name for p in here.rglob("test_*.py")
                      if p.name != pathlib.Path(__file__).name
                      and f"mark.{BLENDER_SMOKE_MARK}" in p.read_text(encoding="utf-8"))
    assert carriers == ["test_atlas_blender_smoke.py"]
