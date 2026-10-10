# leed_daylight/drivers/init.py
"""The ``init`` verb (CLI spec 4.1): start a new project folder.

    leed-daylight init "<project folder>"

It writes ``project.yaml`` from the shipped template (every value
``FILL_ME``, which ``load_project`` refuses until a person fills it in),
copies the shipped LEED v4.1 credit rules template to ``credit_rules.yaml``
byte for byte, and makes empty ``model/`` and ``weather/`` folders. It
never overwrites anything: a folder that already holds one of the four
config files is refused by name. No value in the template is a default
(parent design section 12 rule 3).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections.abc import Sequence

from ..engine.jobs import write_new
from ._folder import REPO_ROOT, check_outside_repo

#: The shipped templates, beside this package's own modules
#: (``pyproject.toml`` ships ``templates/*.yaml`` as package data).
TEMPLATES = pathlib.Path(__file__).resolve().parents[1] / "templates"
PROJECT_TEMPLATE = TEMPLATES / "project_template.yaml"
CREDIT_RULES_TEMPLATE = TEMPLATES / "credit_rules_v41_bdc_nc.yaml"

#: The four config files of a project folder; init refuses a folder that
#: holds any of them.
CONFIG_FILES = ("project.yaml", "materials.csv", "spaces.csv", "credit_rules.yaml")


class InitError(ValueError):
    """The init command cannot proceed."""


def _check_target(folder: pathlib.Path) -> None:
    check_outside_repo(folder, REPO_ROOT, InitError)
    if folder.exists() and not folder.is_dir():
        raise InitError(f"{folder}: not a folder")
    if not folder.exists() and not folder.parent.is_dir():
        raise InitError(f"{folder}: its parent folder {folder.parent} does not exist")
    for name in CONFIG_FILES:
        if os.path.lexists(folder / name):
            raise InitError(f"{folder}: already holds {name}; init never overwrites a project")
    # M3: refuse before anything is written -- a file named `model` or
    # `weather` would otherwise fail the later mkdir only after
    # project.yaml and credit_rules.yaml were already written, leaving a
    # half-made folder that a rerun then refuses ("already holds
    # project.yaml").
    for name in ("model", "weather"):
        target = folder / name
        # lexists, not exists: a dangling link is refused too, never followed.
        if os.path.lexists(target) and (target.is_symlink() or not target.is_dir()):
            raise InitError(f"{target}: exists and is not a plain folder")


def init_folder(folder, *, out=None) -> pathlib.Path:
    """Write a new project folder at ``folder`` and return its path.
    ``out=None`` prints nothing."""
    folder = pathlib.Path(folder)
    _check_target(folder)
    folder.mkdir(exist_ok=True)
    written = []
    for name, source in (("project.yaml", PROJECT_TEMPLATE), ("credit_rules.yaml", CREDIT_RULES_TEMPLATE)):
        target = folder / name
        write_new(target, source.read_bytes(), leftover=False)
        written.append(target)
    for name in ("model", "weather"):
        (folder / name).mkdir(exist_ok=True)
        written.append(folder / name)
    if out is not None:
        for path in written:
            print(f"wrote {path}", file=out)
        print("next:", file=out)
        print(f"  1. put the Radiance scene in {folder / 'model'} and the .wea file in {folder / 'weather'}", file=out)
        print(f"  2. fill every FILL_ME in {folder / 'project.yaml'}", file=out)
        print(f'  3. run: leed-daylight scan "{folder}"', file=out)
    return folder


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.init",
        description="Make a new project folder with a project.yaml to fill in.",
    )
    parser.add_argument("folder", help="the new project folder (made if missing; never overwritten)")
    args = parser.parse_args(argv)
    try:
        init_folder(args.folder, out=sys.stdout)
    except (ValueError, OSError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
