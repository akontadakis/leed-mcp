# leed_daylight/drivers/calculator.py
"""The ``calculator`` command (calculator spec 5.1): the USGBC "LEED v4.1
Daylight and Quality Views Calculator" V02, filled from a case's results.

    leed-daylight calculator "<project folder>" [--case <name>] --template <blank .xlsm>

It writes ``<folder>/_run/calculator/<case>/`` and prints the file, the
rows written, the Summary values Excel should show once it recalculates
the file, and every warning. Exit code 0 when written, 1 on a refusal."""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..deliver.calculator.build import CalculatorResult, build_calculator
from ..deliver.data import load_case
from ..engine.jobs import folder_lock
from ._folder import REPO_ROOT, check_folder

_LABELS = (("total_area", "total regularly occupied area"), ("average_sda", "average sDA300/50% (%)"),
           ("daylit_area", "daylit area (Option 2)"), ("daylit_pct", "daylit share (%)"),
           ("views_area", "area with quality views"), ("views_pct", "quality views share (%)"))


class CalculatorCommandError(ValueError):
    """The calculator command cannot proceed from these inputs."""


def calculator_folder(folder, case: str | None = None, *, template, out=None) -> CalculatorResult:
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    check_folder(folder, REPO_ROOT, CalculatorCommandError)
    result = build_calculator(load_case(folder, case), pathlib.Path(template))
    print(f"wrote {result.path}", file=out)
    print(f"rows: Daylight {result.rows['daylight']}, Quality Views {result.rows['views']}", file=out)
    print("expected Summary values once Excel recalculates:", file=out)
    for key, label in _LABELS:
        value = result.expected.get(key)
        print(f"  {label}: {'-' if value is None else value}", file=out)
    for warning in result.warnings:
        print(f"warning: {warning}", file=out)
    return result


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.calculator",
        description="Fill the USGBC LEED v4.1 Daylight and Quality Views Calculator (V02) from a case.",
    )
    parser.add_argument("folder", help="the project folder")
    parser.add_argument("--case", default=None, help="the case (default: the first case in spaces.csv)")
    parser.add_argument("--template", required=True, help="a blank copy of the calculator (.xlsm)")
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, CalculatorCommandError)
        with folder_lock(pathlib.Path(args.folder), "calculator"):
            calculator_folder(args.folder, args.case, template=args.template)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
