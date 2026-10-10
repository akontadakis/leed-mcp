# tests/test_check_script.py
"""Tests for tools/check.py, the local check that replaces CI (CLI spec 6).

The summary is read from a small invented JUnit XML file; ``main`` is run
with a stand-in for pytest, so this test never runs the suite inside the
suite.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("check_script", ROOT / "tools" / "check.py")
check = importlib.util.module_from_spec(_spec)
# A dataclass looks its own module up in sys.modules while it is built.
sys.modules["check_script"] = check
_spec.loader.exec_module(check)

JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="7">
<testcase classname="tests.test_units" name="test_a"/>
<testcase classname="tests.test_units" name="test_b"/>
<testcase classname="tests.test_scoring" name="test_c"><failure message="boom"/></testcase>
<testcase classname="tests.test_scoring" name="test_d"><error message="bad fixture"/></testcase>
<testcase classname="tests.acceptance.test_fixture_option1" name="test_e"><skipped message="no fixture"/></testcase>
<testcase classname="tests.test_confidentiality" name="test_f"><skipped message="no deny list"/></testcase>
<testcase classname="tests.test_deliver_renderers" name="test_g"><skipped message="no cairo"/></testcase>
</testsuite></testsuites>
"""


def test_summary_counts(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, newline="\n")
    summary = check.summarise(path)
    assert summary == check.Summary(passed=2, failed=2, skipped=3, acceptance_skipped=1,
                                    confidentiality_skipped=1, acceptance_total=1, confidentiality_total=1)


def test_passed_block():
    summary = check.Summary(passed=10, failed=0, skipped=3, acceptance_skipped=1, confidentiality_skipped=1,
                            acceptance_total=1, confidentiality_total=1)
    assert check.format_summary(summary, 0, environ={}) == (
        "CHECK PASSED: 10 passed, 3 skipped, 0 failed\n"
        "  acceptance tests: NOT RUN (1 skipped; set LEED_ACCEPTANCE_FIXTURE)\n"
        "  confidentiality scan: NOT RUN (1 skipped)\n"
        "  other skips: 1"
    )


def test_failed_block_and_tests_that_ran():
    summary = check.Summary(passed=10, failed=1, skipped=0, acceptance_skipped=0, confidentiality_skipped=0,
                            acceptance_total=1, confidentiality_total=1)
    assert check.format_summary(summary, 1, environ={}) == (
        "CHECK FAILED: 10 passed, 0 skipped, 1 failed\n"
        "  acceptance tests: ran (1)\n"
        "  confidentiality scan: ran (1)\n"
        "  other skips: 0"
    )


def test_not_collected_block(tmp_path):
    """I2: a group with no cases collected at all (e.g. ``tools/check.py
    tests/test_units.py``) must say so, rather than the false green 'ran'
    that a skip count of 0 used to print."""
    path = tmp_path / "junit.xml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest" tests="2">\n'
        '<testcase classname="tests.test_units" name="test_a"/>\n'
        '<testcase classname="tests.test_scoring" name="test_b"><failure message="boom"/></testcase>\n'
        '</testsuite></testsuites>\n'
    , newline="\n")
    summary = check.summarise(path)
    assert summary == check.Summary(passed=1, failed=1, skipped=0, acceptance_skipped=0,
                                    confidentiality_skipped=0, acceptance_total=0, confidentiality_total=0)
    assert check.format_summary(summary, 1, environ={}) == (
        "CHECK FAILED: 1 passed, 0 skipped, 1 failed\n"
        "  acceptance tests: NOT RUN (not collected)\n"
        "  confidentiality scan: NOT RUN (not collected)\n"
        "  other skips: 0"
    )


def test_some_run_and_some_skipped_keeps_the_not_run_wording(tmp_path):
    """I2: skips win -- a group with at least one skip is still NOT RUN,
    even if other cases in the same group ran."""
    path = tmp_path / "junit.xml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest" tests="2">\n'
        '<testcase classname="tests.acceptance.test_fixture_option1" name="test_a"/>\n'
        '<testcase classname="tests.acceptance.test_fixture_option1" name="test_b">'
        '<skipped message="no fixture"/></testcase>\n'
        '</testsuite></testsuites>\n'
    , newline="\n")
    summary = check.summarise(path)
    assert summary.acceptance_total == 2
    assert summary.acceptance_skipped == 1
    assert check.format_summary(summary, 0, environ={}).splitlines()[1] == (
        "  acceptance tests: NOT RUN (1 skipped; set LEED_ACCEPTANCE_FIXTURE)"
    )


NO_PASS_MARK_JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3">
<testcase classname="tests.acceptance.test_fixture_option1" name="test_a"/>
<testcase classname="tests.acceptance.test_fixture_option1" name="test_sda">\
<skipped message="run 'B' has no sDA target (no pass mark); got 50.000"/></testcase>
<testcase classname="tests.acceptance.test_fixture_option1" name="test_ase">\
<skipped message="run 'B' has no ASE target (no pass mark); got 20.000"/></testcase>
</testsuite></testsuites>
"""


def test_with_the_fixture_set_a_run_without_a_pass_mark_says_so(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(NO_PASS_MARK_JUNIT, newline="\n")
    summary = check.summarise(path)

    assert summary.acceptance_no_pass_mark == 2
    line = check.format_summary(summary, 0, environ={"LEED_ACCEPTANCE_FIXTURE": "/fixture"}).splitlines()[1]
    assert line == "  acceptance tests: NOT RUN (2 skipped; the run has no pass mark)"
    assert check.format_summary(summary, 0, environ={}).splitlines()[1] == (
        "  acceptance tests: NOT RUN (2 skipped; set LEED_ACCEPTANCE_FIXTURE)"
    )


def test_with_the_fixture_set_another_skip_points_at_the_reasons(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, newline="\n")
    summary = check.summarise(path)

    line = check.format_summary(summary, 0, environ={"LEED_ACCEPTANCE_FIXTURE": "/fixture"}).splitlines()[1]
    assert line == "  acceptance tests: NOT RUN (1 skipped; LEED_ACCEPTANCE_FIXTURE is set, see the skip reasons)"


def test_main_passes_arguments_and_returns_the_exit_code(capsys):
    seen = []

    def fake_pytest(args):
        seen.append(args)
        junit = pathlib.Path(args[args.index("--junitxml") + 1])
        junit.write_text(JUNIT, newline="\n")
        return 1

    assert check.main(["-x", "tests/test_units.py"], run_pytest=fake_pytest) == 1
    (args,) = seen
    assert args[-2:] == ["-x", "tests/test_units.py"]
    assert capsys.readouterr().out.splitlines()[-4].startswith("CHECK FAILED: 2 passed, 3 skipped, 2 failed")


def test_no_report_when_pytest_did_not_start(capsys):
    assert check.main([], run_pytest=lambda args: 4) == 4
    assert "CHECK FAILED: pytest wrote no report (exit code 4)" in capsys.readouterr().out
