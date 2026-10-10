#!/usr/bin/env python3
# tools/check.py
"""The local check (CLI spec 6): the whole test suite, then one clear verdict.

    .venv/bin/python tools/check.py [pytest arguments]

CI runs this where the public copy is published, on macOS, Windows and Linux.
This runs ``python -m pytest`` from the repository root with a JUnit XML
report, streams pytest's own output, then prints one block: CHECK PASSED or
CHECK FAILED with the counts, and whether the acceptance tests (they need
a fixture outside the repository) and the confidentiality scan ran or were
skipped. It returns pytest's exit
code. A developer script: not part of the ``leed-daylight`` command and not
shipped (``pyproject.toml`` installs only ``leed_daylight*``).
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_ENV_VAR = "LEED_ACCEPTANCE_FIXTURE"
#: The words a gate test's skip reason carries for a run without a target
#: (``tests/acceptance/test_fixture_option1.py``; ``tests/conftest.py``).
NO_PASS_MARK = "(no pass mark)"


@dataclasses.dataclass(frozen=True)
class Summary:
    passed: int
    failed: int
    skipped: int
    acceptance_skipped: int
    confidentiality_skipped: int
    acceptance_total: int
    confidentiality_total: int
    #: Acceptance skips whose reason is a run without a pass mark.
    acceptance_no_pass_mark: int = 0


def summarise(junit: pathlib.Path) -> Summary:
    """Count the test cases of a pytest JUnit XML report. An error counts as
    a failure. ``acceptance_total``/``confidentiality_total`` count every
    case collected for that group, run or not -- a group with nothing
    collected at all (e.g. ``-k`` or a single file that has none of its
    cases) is not the same as one that ran and passed (I2)."""
    passed = failed = skipped = acceptance_skipped = confidentiality_skipped = 0
    acceptance_total = confidentiality_total = acceptance_no_pass_mark = 0
    for case in ET.parse(junit).getroot().iter("testcase"):
        tags = {child.tag for child in case}
        classname = case.get("classname", "")
        is_acceptance = classname.startswith("tests.acceptance")
        is_confidentiality = classname == "tests.test_confidentiality"
        acceptance_total += is_acceptance
        confidentiality_total += is_confidentiality
        if tags & {"failure", "error"}:
            failed += 1
        elif "skipped" in tags:
            skipped += 1
            if is_acceptance:
                acceptance_skipped += 1
                acceptance_no_pass_mark += NO_PASS_MARK in (case.find("skipped").get("message") or "")
            elif is_confidentiality:
                confidentiality_skipped += 1
        else:
            passed += 1
    return Summary(passed, failed, skipped, acceptance_skipped, confidentiality_skipped,
                    acceptance_total, confidentiality_total, acceptance_no_pass_mark)


def _group_line(skipped: int, total: int, note: str) -> str:
    """``skipped`` wins over ``total`` (a group with some run and some
    skipped keeps the NOT RUN wording); no cases collected at all is its
    own, distinct wording -- not the false green a skip count of 0 used to
    print (I2)."""
    if skipped:
        return f"NOT RUN ({skipped} skipped{note})"
    if not total:
        return "NOT RUN (not collected)"
    return f"ran ({total})"


def _acceptance_note(summary: Summary, environ: Mapping[str, str]) -> str:
    """Why the acceptance tests skipped, as ``tests/conftest.py``'s banner
    says it: only an unset fixture variable asks for it."""
    if not environ.get(FIXTURE_ENV_VAR):
        return f"; set {FIXTURE_ENV_VAR}"
    if summary.acceptance_no_pass_mark == summary.acceptance_skipped:
        return "; the run has no pass mark"
    return f"; {FIXTURE_ENV_VAR} is set, see the skip reasons"


def format_summary(summary: Summary, exit_code: int, environ: Mapping[str, str] | None = None) -> str:
    """The closing block; ``environ`` (default ``os.environ``) is read only
    for the acceptance line's note."""
    environ = os.environ if environ is None else environ
    verdict = "CHECK PASSED" if exit_code == 0 else "CHECK FAILED"
    acceptance = _group_line(
        summary.acceptance_skipped, summary.acceptance_total, _acceptance_note(summary, environ),
    )
    confidentiality = _group_line(summary.confidentiality_skipped, summary.confidentiality_total, "")
    other = summary.skipped - summary.acceptance_skipped - summary.confidentiality_skipped
    return (
        f"{verdict}: {summary.passed} passed, {summary.skipped} skipped, {summary.failed} failed\n"
        f"  acceptance tests: {acceptance}\n"
        f"  confidentiality scan: {confidentiality}\n"
        f"  other skips: {other}"
    )


def _run_pytest(args: Sequence[str]) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", *args], cwd=ROOT).returncode


def main(argv: Sequence[str] | None = None,
         run_pytest: Callable[[Sequence[str]], int] = _run_pytest) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    with tempfile.TemporaryDirectory() as tmp:
        junit = pathlib.Path(tmp) / "junit.xml"
        code = run_pytest(["--junitxml", str(junit), *argv])
        print()
        if not junit.is_file():
            print(f"CHECK FAILED: pytest wrote no report (exit code {code})")
            return code
        print(format_summary(summarise(junit), code))
    return code


if __name__ == "__main__":
    sys.exit(main())
