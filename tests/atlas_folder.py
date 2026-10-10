# tests/atlas_folder.py
"""Shared helpers for the atlas tests (atlas plan Tasks 4 and 6).

``atlas_inputs`` loads a case as the atlas does (every option,
``require_results=False``), prepares its model into ``work`` and builds
the bundle. ``edit_points`` rewrites one option's points.csv.
``millimetre_copy`` copies a project folder with its scene redrawn in
millimetres. Every number is invented and round.
"""
from __future__ import annotations

import csv
import pathlib
import re
import shutil

import yaml

from leed_daylight.deliver.atlas.bundle import build_bundle
from leed_daylight.deliver.data import load_case, prepare_case_model

CASE = "base_case"


def atlas_inputs(folder, work, case=None):
    """(case data, prepared model, bundle) of ``folder``, prepared into ``work``."""
    data = load_case(folder, case, require_results=False)
    prepared = prepare_case_model(data.project, data.spaces, data.case, pathlib.Path(work)).prepared
    return data, prepared, build_bundle(data, prepared)


def edit_points(folder, option: str, change) -> None:
    """Read ``_run/<option>/base_case/points.csv`` as a header and rows,
    pass both through ``change`` (edits in place), write them back."""
    path = pathlib.Path(folder) / "_run" / option / CASE / "points.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    change(header, body)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)


_POLYGON_REALS = re.compile(r"^12\s")


def millimetre_copy(folder, target) -> pathlib.Path:
    """A copy of ``folder`` at ``target`` whose scene.rad is drawn in
    millimetres (every polygon coordinate times 1000) and whose
    project.yaml says ``scene.units: mm``."""
    target = pathlib.Path(target)
    shutil.copytree(folder, target)
    lines = []
    for line in (target / "scene.rad").read_text().splitlines():
        if _POLYGON_REALS.match(line):
            words = line.split()
            line = " ".join([words[0], *(f"{float(w) * 1000:g}" for w in words[1:])])
        lines.append(line)
    (target / "scene.rad").write_text("\n".join(lines) + "\n")
    raw = yaml.safe_load((target / "project.yaml").read_text())
    raw["scene"]["units"] = "mm"
    (target / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return target
