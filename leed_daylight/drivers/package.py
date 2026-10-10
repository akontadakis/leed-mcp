# leed_daylight/drivers/package.py
"""The ``package`` command (reports spec 9): the submission folder of one case.

    python -m leed_daylight.drivers.package "<project folder>" [--case <name>]

Run the report command first; the package refuses reports that are missing
or older than their results, or an option1 report whose glare narrative has
since changed on disk or was made for a different case (pre-flight ruling
F6), or a floor atlas that is missing or out of date (atlas spec 11; reports
made before the atlas was added have none, so ``report`` runs once more).

Pre-flight ruling F1: ``out`` defaults to ``None`` and is resolved to
``sys.stdout`` *inside* this function, not at import time -- the same
reasoning as ``drivers/report.py``'s ``report_folder``: an import-time
default is bound once, before a test's ``capsys`` (or any other stdout
replacement) has taken effect.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..deliver.data import load_case
from ..deliver.package import CHECKSUMS, build_package
from ..engine.jobs import folder_lock
from ._folder import REPO_ROOT, check_folder


class PackageCommandError(ValueError):
    """The package command cannot proceed from these inputs."""


def package_folder(folder, case: str | None = None, *, out=None,
                   now: datetime.datetime | None = None) -> pathlib.Path:
    folder = pathlib.Path(folder)
    out = sys.stdout if out is None else out
    check_folder(folder, REPO_ROOT, PackageCommandError)
    now = now or datetime.datetime.now()
    data = load_case(folder, case, today=now.date())
    path = build_package(data, submission_root=folder / "submission", now=now, out=out)
    for option, reason in data.skipped.items():
        print(f"left out {option}: {reason}", file=out)
    print(f"package written to {path}", file=out)
    print(f"check it with: cd \"{path}\" && shasum -a 256 -c {CHECKSUMS}", file=out)
    return path


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.package",
        description="Write the checksummed submission folder of one case.",
    )
    parser.add_argument("folder", help="the project folder")
    parser.add_argument("--case", default=None, help="the case (default: the first case in spaces.csv)")
    args = parser.parse_args(argv)
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, PackageCommandError)
        with folder_lock(pathlib.Path(args.folder), "package"):
            package_folder(args.folder, args.case)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
