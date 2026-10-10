# tests/conftest.py
"""Shared test fixtures.

``write_example``/``example_project_dir`` give every ``test_project_config``
case its own copy of the example project, in its own ``tmp_path``
subdirectory, with every file the yaml references already created on disk
(an empty scene file, a valid fictional 8760-hour weather file, a minimal
credit_rules.yaml). Tests never load
the fixture in place: a future task will make ``load_project`` check that
referenced files exist, and these tests must not need rewriting when it does.
The two factories are also plain functions (``write_example_project``,
``write_wea_file``), for helpers that run outside pytest.

No test starts a real Blender by accident (atlas spec 15): every test runs
with ``BLENDER`` set to ``NO_BLENDER``, a path that does not exist, except
the opt-in smoke test (marked ``blender_smoke``) when it is asked for with
``LEED_BLENDER_SMOKE=1``. The variable never lifts the guard for any other
test (final review M2).

No test opens a real browser (wizard plan, Global Constraints): every
``webbrowser`` opener (``open``, ``open_new``, ``open_new_tab`` and
``get``) is replaced, for every test, by one that fails the test with
``a test tried to open a real browser``. The wizard verb takes its opener
as an argument, so its own tests hand it a recorder instead. No test reads
or writes the real ``~/.leed-daylight/``: the wizard's recent-projects file
is pointed into a temporary folder of its own for every test. No test
signals a real process group (``os.killpg``, and ``taskkill`` on Windows,
fail the test), so stopping a fake job can never reach pytest's own process.
"""
from __future__ import annotations

import calendar
import functools
import os
import pathlib
import sys
import webbrowser
from collections.abc import Mapping

import pytest
import yaml

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

#: The environment variable that asks for the opt-in Blender smoke run.
BLENDER_SMOKE_ENV = "LEED_BLENDER_SMOKE"
#: What ``BLENDER`` is set to in every other test: a path that never
#: exists. The Blender finder uses ``BLENDER`` whenever it is set and
#: refuses a path that is not there; it never falls back to ``PATH`` or
#: /Applications, so no test can find a Blender installed on the machine.
NO_BLENDER = FIXTURES / "no_blender_here" / "blender"


#: The mark of the opt-in Blender smoke test (tests/test_atlas_blender_smoke.py).
BLENDER_SMOKE_MARK = "blender_smoke"


def pytest_configure(config):
    config.addinivalue_line("markers", f"{BLENDER_SMOKE_MARK}: the opt-in Blender smoke test")


def blender_for_tests(environ: Mapping[str, str], *, smoke_test: bool) -> str | None:
    """The ``BLENDER`` value a test runs with: ``NO_BLENDER``; or None
    (leave the environment as it is) only for the opt-in smoke test
    (``smoke_test``) when ``environ`` asks for the smoke run."""
    if smoke_test and environ.get(BLENDER_SMOKE_ENV) == "1":
        return None
    return str(NO_BLENDER)


@pytest.fixture(autouse=True)
def _no_real_blender(request, monkeypatch):
    smoke_test = request.node.get_closest_marker(BLENDER_SMOKE_MARK) is not None
    value = blender_for_tests(os.environ, smoke_test=smoke_test)
    if value is not None:
        monkeypatch.setenv("BLENDER", value)


#: The ``webbrowser`` names that open or hand out a real browser.
BROWSER_OPENERS = ("open", "open_new", "open_new_tab", "get")


def _refuse_real_browser(*args, **kwargs):
    pytest.fail("a test tried to open a real browser")


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    for name in BROWSER_OPENERS:
        monkeypatch.setattr(webbrowser, name, _refuse_real_browser)


def _refuse_real_signal(*args, **kwargs):
    pytest.fail("a test tried to signal a real process group")


@pytest.fixture(autouse=True)
def _no_real_job_signal(monkeypatch):
    """A job's fake process carries the test's own process id, so the real
    stop (``os.killpg``) would signal pytest itself: it fails the test
    instead; on Windows the real stop is ``taskkill``, refused the same way.
    Tests stop jobs with the fake signaller of ``job_fakes``."""
    monkeypatch.setattr(os, "killpg", _refuse_real_signal, raising=False)
    from leed_daylight.engine import jobs

    monkeypatch.setattr(jobs._TaskkillSignaller, "_taskkill", staticmethod(_refuse_real_signal))


#: Windows: the error for a link made without the create-symlink right.
ERROR_PRIVILEGE_NOT_HELD = 1314


def symlink_or_skip(link, target):
    """Make ``link`` point at ``target``, or skip the test where this system
    does not allow it (Windows without the create-symlink right,
    ``ERROR_PRIVILEGE_NOT_HELD``); any other failure fails the test. A
    relative ``target`` is taken from ``link``'s folder, as the link will."""
    target_dir = (pathlib.Path(link).parent / target).is_dir()
    try:
        link.symlink_to(target, target_is_directory=target_dir)
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == ERROR_PRIVILEGE_NOT_HELD:
            pytest.skip(f"cannot create a link here: {exc}")
        raise


@pytest.fixture(autouse=True)
def _no_real_recent_file(monkeypatch, tmp_path_factory):
    """``files.default_recent_file`` gives one file in a temporary folder,
    made the first time a test asks for it."""
    from leed_daylight.edit import files

    made: list[pathlib.Path] = []

    def recent_file() -> pathlib.Path:
        if not made:
            made.append(tmp_path_factory.mktemp("home") / ".leed-daylight" / "recent.json")
        return made[0]

    monkeypatch.setattr(files, "default_recent_file", recent_file)


@pytest.fixture(autouse=True)
def _no_real_job_index(monkeypatch, tmp_path_factory):
    """``LEED_MCP_STATE`` names a folder in the test run's temporary folder,
    so a server built from the environment never touches ``~/.leed-mcp``
    (leed-mcp spec 8.1). Nothing is created until a job index is used."""
    monkeypatch.setenv("LEED_MCP_STATE", str(tmp_path_factory.getbasetemp() / "leed-mcp-state"))


#: A minimal, valid, fictional credit_rules.yaml -- shape only, never scored
#: by these tests. See leed_daylight/project/rules.py for the real loader.
MINIMAL_CREDIT_RULES = """\
# Minimal fictional rules, for tests only.
version: "Example rules v0"
max_points: 3
option1:
  sda:
    lux: 300
    fraction: 0.50
    bands:
      - {min_pct: 40.0, points: 1}
  ase:
    lux: 1000
    hours: 250
    per_space_narrative_trigger_pct: 10.0
option2:
  bands: UNVERIFIED
  lux_min: 300
  lux_max: 3000
glare_control_required: true
"""


@functools.lru_cache(maxsize=None)
def _full_year_lines() -> tuple[str, ...]:
    return tuple(
        f"{month} {day} {hour + 0.5} 0 0"
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(2023, month)[1] + 1)
        for hour in range(24)
    )


def wea_text(latitude=42.0, longitude_W=80.0, time_zone=75.0, hours=None):
    """A valid, fictional Radiance .wea: header plus one line per hour.

    The six header lines ``gendaymtx`` reads, then 8760 data lines
    ``month day hour direct diffuse`` in a non-leap year, each hour stamp
    centred on its half hour (0.5 ... 23.5), with zero irradiance. The
    location defaults match ``fixtures/project_example.yaml`` (42 N, 80 W
    -- .wea longitudes are west-positive). ``hours`` replaces the data
    lines, for tests that break them on purpose.
    """
    header = (
        "place Example\n"
        f"latitude {latitude}\n"
        f"longitude {longitude_W}\n"
        f"time_zone {time_zone}\n"
        "site_elevation 0.0\n"
        "weather_data_file_units 1\n"
    )
    if hours is None:
        hours = _full_year_lines()
    return header + "\n".join(hours) + "\n"


def _wea_site(raw) -> dict:
    """The mutated project's own site, as ``wea_text`` keywords, so the
    weather file always agrees with project.yaml (never a spurious
    site-mismatch warning); the defaults when the site is malformed."""
    try:
        latitude = float(raw["site"]["latitude_N"])
        longitude_W = -float(raw["site"]["longitude_E"])
    except (KeyError, TypeError, ValueError):
        return {}
    return {"latitude": latitude, "longitude_W": longitude_W}


def write_wea_file(path, **kwargs):
    """Write ``wea_text(**kwargs)`` to ``path``; returns the path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wea_text(**kwargs))
    return path


@pytest.fixture
def write_wea():
    """Factory: ``write_wea(path, **kwargs)`` writes ``wea_text(**kwargs)``."""
    return write_wea_file


def write_example_project(tmp_path, mutate=lambda d: d):
    """Write a mutated copy of project_example.yaml to a fresh dir.

    Loads the example yaml into a dict, passes it through ``mutate`` (which
    may edit the dict in place and/or return the dict to write), writes the
    result as ``project.yaml`` in a fresh subdirectory of ``tmp_path``, and
    creates every file the (unmutated) example references alongside it.
    Returns the path to the written project.yaml.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    raw = yaml.safe_load((FIXTURES / "project_example.yaml").read_text())
    raw = mutate(raw)

    project_path = project_dir / "project.yaml"
    project_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    (project_dir / "scene.rad").write_text("")
    weather_dir = project_dir / "weather"
    weather_dir.mkdir()
    (weather_dir / "example.wea").write_text(wea_text(**_wea_site(raw)))
    (project_dir / "credit_rules.yaml").write_text(MINIMAL_CREDIT_RULES)

    return project_path


@pytest.fixture
def write_example():
    """Factory: ``write_example(tmp_path, mutate=lambda d: d)`` is
    :func:`write_example_project`."""
    return write_example_project


@pytest.fixture
def example_project_dir(tmp_path, write_example):
    """A fresh, unmutated copy of the example project directory."""
    return write_example(tmp_path).parent


#: The line printed after the summary whenever the acceptance gate skipped.
ACCEPTANCE_NOT_RUN = "ACCEPTANCE GATE NOT RUN: set LEED_ACCEPTANCE_FIXTURE"
#: The same line when the fixture variable is set and every acceptance skip
#: is a run without a pass mark (a gate test of a run with no target).
ACCEPTANCE_NO_PASS_MARK = "ACCEPTANCE GATE NOT RUN: the run has no pass mark"
#: The same line when the fixture variable is set and some skip has another reason.
ACCEPTANCE_FIXTURE_SET = "ACCEPTANCE GATE NOT RUN: LEED_ACCEPTANCE_FIXTURE is set; see the skip reasons"
#: The words a gate test's skip reason carries for a run without a target.
NO_PASS_MARK = "(no pass mark)"


def acceptance_banner(skipped, environ) -> str | None:
    """The banner line for the skipped test reports ``skipped``, or None
    when no ``acceptance``-marked test is among them. ``environ`` is the
    environment to read (``os.environ`` in a session; a plain mapping in
    a test): only an unset ``LEED_ACCEPTANCE_FIXTURE`` asks for it."""
    reasons = [str(getattr(report, "longrepr", "")) for report in skipped
               if "acceptance" in getattr(report, "keywords", {})]
    if not reasons:
        return None
    if not environ.get("LEED_ACCEPTANCE_FIXTURE"):
        line = ACCEPTANCE_NOT_RUN
    elif all(NO_PASS_MARK in reason for reason in reasons):
        line = ACCEPTANCE_NO_PASS_MARK
    else:
        line = ACCEPTANCE_FIXTURE_SET
    return f"{line} ({len(reasons)} acceptance test(s) skipped)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say loudly when any ``acceptance``-marked test was skipped.

    Without the private fixture every acceptance test skips, and a green
    run then looks exactly like a run that passed the acceptance gate. This
    prints one unmistakable line so it cannot be mistaken for one. With the
    fixture set, a run without a pass mark says so instead
    (``acceptance_banner``).
    """
    banner = acceptance_banner(terminalreporter.stats.get("skipped", []), os.environ)
    if banner:
        terminalreporter.write_sep("=", "acceptance", yellow=True, bold=True)
        terminalreporter.write_line(banner, yellow=True, bold=True)
