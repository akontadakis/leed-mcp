# tests/test_driver_prog.py
"""Each existing command's ``main`` takes an optional ``prog`` (CLI spec 3.4).

``leed-daylight <verb>`` passes ``prog="leed-daylight <verb>"`` so the
usage line names the command the person typed. Without it, the usage line
is the old ``python -m leed_daylight.drivers.<name>`` one, unchanged.
"""
from __future__ import annotations

import importlib

import pytest

NAMES = ("rooms", "run", "option2", "views", "report", "atlas", "package")


@pytest.mark.parametrize("name", NAMES)
def test_prog_names_the_usage_line(name, capsys):
    module = importlib.import_module(f"leed_daylight.drivers.{name}")
    with pytest.raises(SystemExit) as info:
        module.main(["--help"], prog=f"leed-daylight {name}")
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: leed-daylight {name} ")


@pytest.mark.parametrize("name", NAMES)
def test_without_prog_the_old_usage_line_stays(name, capsys):
    module = importlib.import_module(f"leed_daylight.drivers.{name}")
    with pytest.raises(SystemExit):
        module.main(["--help"])
    assert capsys.readouterr().out.startswith(f"usage: python -m leed_daylight.drivers.{name} ")
