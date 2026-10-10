# tests/test_acceptance_gate_warning.py
"""A skipped acceptance gate must be loud, not a quiet 's' in the dots.

``tests/conftest.py``'s ``pytest_terminal_summary`` prints one warning
line when any ``acceptance``-marked test was skipped. Driven here with
fake reports, so this test never depends on whether the fixture is set.
"""
from __future__ import annotations

import pathlib
import types

import pytest
from conftest import (
    ACCEPTANCE_FIXTURE_SET,
    ACCEPTANCE_NO_PASS_MARK,
    ACCEPTANCE_NOT_RUN,
    NO_PASS_MARK,
    acceptance_banner,
    pytest_terminal_summary,
)

FIXTURE_VAR = "LEED_ACCEPTANCE_FIXTURE"


class _FakeReporter:
    def __init__(self, skipped):
        self.stats = {"skipped": skipped} if skipped else {}
        self.lines: list[str] = []

    def write_line(self, line, **markup):
        self.lines.append(line)

    def write_sep(self, sep, title="", **markup):
        self.lines.append(title)


def _report(*keywords, reason="Skipped: no fixture"):
    return types.SimpleNamespace(keywords={k: 1 for k in keywords}, longrepr=("test_x.py", 1, reason))


@pytest.fixture(autouse=True)
def _no_fixture_variable(monkeypatch):
    monkeypatch.delenv(FIXTURE_VAR, raising=False)


def test_a_skipped_acceptance_test_prints_the_warning():
    reporter = _FakeReporter([_report("test_x", "acceptance"), _report("test_y")])
    pytest_terminal_summary(reporter, 0, None)
    assert any(ACCEPTANCE_NOT_RUN in line for line in reporter.lines)
    assert ACCEPTANCE_NOT_RUN == "ACCEPTANCE GATE NOT RUN: set LEED_ACCEPTANCE_FIXTURE"


def test_no_warning_when_no_acceptance_test_was_skipped():
    reporter = _FakeReporter([_report("test_y")])
    pytest_terminal_summary(reporter, 0, None)
    assert reporter.lines == []

    reporter = _FakeReporter([])
    pytest_terminal_summary(reporter, 0, None)
    assert reporter.lines == []


NO_PASS_MARK_REASON = "Skipped: run 'B' has no sDA target (no pass mark); got 50.000"


def test_without_the_variable_the_banner_asks_for_it():
    banner = acceptance_banner([_report("acceptance", reason=NO_PASS_MARK_REASON)], {})

    assert banner == f"{ACCEPTANCE_NOT_RUN} (1 acceptance test(s) skipped)"


def test_with_the_variable_set_a_run_without_a_pass_mark_says_so():
    reports = [_report("acceptance", reason=NO_PASS_MARK_REASON)] * 2

    banner = acceptance_banner(reports, {FIXTURE_VAR: "/fixture"})

    assert banner == f"{ACCEPTANCE_NO_PASS_MARK} (2 acceptance test(s) skipped)"
    assert "set LEED_ACCEPTANCE_FIXTURE" not in banner


def test_with_the_variable_set_any_other_skip_points_at_the_reasons():
    reports = [_report("acceptance", reason=NO_PASS_MARK_REASON), _report("acceptance")]

    banner = acceptance_banner(reports, {FIXTURE_VAR: "/fixture"})

    assert banner == f"{ACCEPTANCE_FIXTURE_SET} (2 acceptance test(s) skipped)"
    assert "set LEED_ACCEPTANCE_FIXTURE" not in banner


def test_the_terminal_summary_reads_the_variable(monkeypatch):
    monkeypatch.setenv(FIXTURE_VAR, "/fixture")
    reporter = _FakeReporter([_report("acceptance", reason=NO_PASS_MARK_REASON)])

    pytest_terminal_summary(reporter, 0, None)

    assert any(ACCEPTANCE_NO_PASS_MARK in line for line in reporter.lines)


def test_no_banner_without_an_acceptance_skip():
    assert acceptance_banner([_report("test_y")], {FIXTURE_VAR: "/fixture"}) is None


def test_the_gate_tests_skip_with_the_words_the_banner_looks_for():
    source = (pathlib.Path(__file__).parent / "acceptance" / "test_fixture_option1.py").read_text(encoding="utf-8")

    assert source.count(NO_PASS_MARK) == 2
