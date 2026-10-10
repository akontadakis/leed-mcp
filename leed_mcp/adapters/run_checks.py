# leed_mcp/adapters/run_checks.py
"""The checks before a job starts (spec 6.4, 6.6, 10), so that most
refusals come from this server, not from a job that fails at once.

In order: the required files are there and not links (``FILE_MISSING``,
``PATH_UNSAFE``); project.yaml has no ``FILL_ME`` left (``FILL_ME_LEFT``,
the keys); every path it names is inside the folder (``PATH_OUTSIDE_ROOTS``,
spec 10, because the file can be edited by hand); every scene file and
``!xform`` target too (``SCENE_INCLUDE_OUTSIDE``); the project
loads (``load_project``, ``INPUT_INVALID`` with the pipeline's message);
the case is one spaces.csv has (``CASE_UNKNOWN``), the first when none is
given, as the CLI does. Then each verb's own: the Radiance programs
(``RADIANCE_NOT_FOUND``), ``processes``, the report libraries
(``REPORT_LIBS_MISSING``) and the results a deliver verb needs
(``RESULTS_MISSING``).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import zipfile
import zlib

import yaml

from leed_daylight.deliver.calculator import v02
from leed_daylight.deliver.calculator.xlsx_rows import Book
from leed_daylight.deliver.data import result_path
from leed_daylight.deliver.manifest import REPORT_MANIFEST
from leed_daylight.project.config import load_project, load_scene_settings
from leed_daylight.project.materials import VLT_ROLES, load_materials, vlt_values
from leed_daylight.project.spaces import case_names, load_spaces, views_case_names
from leed_daylight.scene.input_file import InputFileError
from leed_mcp.adapters.input_files import check_project_paths, read_project_raw
from leed_mcp.adapters.project_state import fill_me_keys, radiance_missing
from leed_mcp.adapters.results_files import CREDITS, refuse_linked
from leed_mcp.adapters.scene import check_inside
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line

__all__ = ["CALCULATOR_ENV", "REPORT_LIBRARIES", "calculator_template", "check_calculator", "check_processes", "check_project", "check_radiance",
           "check_report_libraries", "check_results", "resolve_case"]

#: The report's libraries, by distribution name -> import name: python-docx
#: always, then the renderer's (``deliver.figures.base._PAINTERS``).
REPORT_LIBRARIES = {"python-docx": "docx"}
_RENDERER_LIBRARIES = {"matplotlib": {"matplotlib": "matplotlib"}, "cairosvg": {"cairosvg": "cairosvg"},
                       "pillow": {"Pillow": "PIL"}}
#: The server variable naming the blank USGBC calculator (calculator spec D1).
CALCULATOR_ENV = "LEED_MCP_CALCULATOR"
#: The files every job reads.
_REQUIRED = ("project.yaml", "materials.csv", "spaces.csv")


def check_project(folder: pathlib.Path) -> dict:
    """project.yaml, loaded raw, once every check above the case passed.
    The scene files are checked for every job: ``report`` and ``atlas``
    rebuild the case's model from the scene too."""
    missing = []
    for name in _REQUIRED:
        refuse_linked(folder, folder / name)
        if not (folder / name).is_file():
            missing.append(name)
    if missing:
        raise Refused(refusal("FILE_MISSING", f"{folder}: {', '.join(missing)} missing; a job needs project.yaml, "
                              "materials.csv and spaces.csv (inputs action='create').", field="folder"))
    try:
        _, raw = read_project_raw(folder)
    except InputFileError:
        raise  # not a plain file, or too large: PATH_UNSAFE (errors.MAPPING)
    except (ValueError, yaml.YAMLError) as exc:
        raise Refused(refusal("INPUT_INVALID", f"project.yaml cannot be read: {one_line(exc)}")) from None
    left = fill_me_keys(raw)
    if left:
        raise Refused(refusal("FILL_ME_LEFT", f"project.yaml still has {len(left)} FILL_ME value(s): "
                              f"{', '.join(left)}. Ask the user for each and set it (inputs action='update').",
                              field="folder"))
    check_project_paths(folder, raw)
    check_inside(folder, load_scene_settings(folder / "project.yaml").files)
    load_project(folder / "project.yaml")
    return raw


def resolve_case(folder: pathlib.Path, case: str | None, *, views: bool = False, either: bool = False) -> str:
    """``case``, or the first ``include_<case>`` column (``views``: the
    first ``views_<case>`` column) of spaces.csv; ``CASE_UNKNOWN`` when
    spaces.csv has no such column. ``either`` (not with ``views``) is what
    ``deliver`` takes, as ``deliver.data`` does: the ``include_<case>`` then
    the ``views_<case>`` columns, repeats dropped, the first of them by
    default."""
    spaces = load_spaces(folder / "spaces.csv")
    if either:
        cases = tuple(dict.fromkeys((*case_names(spaces), *views_case_names(spaces))))
        kind = "include_<case> or views_<case>"
    else:
        cases = views_case_names(spaces) if views else case_names(spaces)
        kind = "views_<case>" if views else "include_<case>"
    if not cases:
        raise Refused(refusal("CASE_UNKNOWN", f"spaces.csv has no {kind} column; add a case first "
                              "(inputs action='update', file='spaces', add_case).", field="case"))
    if case is None:
        return cases[0]
    if case not in cases:
        raise Refused(refusal("CASE_UNKNOWN", f"spaces.csv has no {kind} column for {case!r}; its cases are "
                              f"{', '.join(cases)}.", field="case"))
    return case


def check_radiance(action: str) -> None:
    """``RADIANCE_NOT_FOUND`` naming the programs the run ``action`` needs
    that are not on this server's PATH (``project status`` lists the same)."""
    missing = radiance_missing()[action]
    if missing:
        raise Refused(refusal("RADIANCE_NOT_FOUND", f"{', '.join(missing)} not found on this server's PATH; add "
                              "the Radiance bin folder to PATH in the client's server configuration."))


def check_processes(processes: int | None) -> None:
    """``processes`` from 1 to ``os.cpu_count()`` (spec 6.4)."""
    most = os.cpu_count() or 1
    if processes is not None and not 1 <= processes <= most:
        raise Refused(refusal("INPUT_INVALID", f"processes must be from 1 to {most} on this computer; "
                              f"got {processes}.", field="processes"))


def check_report_libraries(raw: dict, renderer: str | None) -> None:
    """``REPORT_LIBS_MISSING`` when python-docx or the renderer's library is
    not installed: found, never imported (spec 3: the extra is optional)."""
    report = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    chosen = renderer or report.get("renderer") or "matplotlib"
    needed = {**REPORT_LIBRARIES, **_RENDERER_LIBRARIES.get(chosen, {})}
    missing = [dist for dist, module in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise Refused(refusal("REPORT_LIBS_MISSING", f"the report needs {', '.join(missing)} (renderer {chosen}), "
                              "which this server's environment lacks; install the report extra "
                              "(uv sync --extra report) and restart the server."))


def check_results(folder: pathlib.Path, verb: str, case: str, option: str | None) -> None:
    """``RESULTS_MISSING``: ``report`` and ``atlas`` need a results file of
    the case (of ``option`` when given); ``package`` needs the case's report."""
    if verb == "package":
        if not (folder / "_run" / "reports" / case / REPORT_MANIFEST).is_file():
            raise Refused(refusal("RESULTS_MISSING", f"case {case!r} has no report yet; run "
                                  "deliver(action='report') first.", field="case"))
        return
    credits = (option,) if option else CREDITS
    if not any(result_path(folder, credit, case).is_file() for credit in credits):
        what = f"{option} results" if option else "results"
        raise Refused(refusal("RESULTS_MISSING", f"case {case!r} has no {what} yet; run the credit first "
                              "(run action='option1', 'option2' or 'views').", field="case"))


#: Templates that passed the full V02 check, by (path, mtime, size): status reads the file on every call.
_CHECKED: set[tuple[str, int, int]] = set()


def calculator_template() -> pathlib.Path:
    """The blank calculator ``LEED_MCP_CALCULATOR`` names, checked as V02;
    ``CALCULATOR_NOT_CONFIGURED`` or ``CALCULATOR_UNSUPPORTED`` otherwise."""
    text = os.environ.get(CALCULATOR_ENV, "")
    path = pathlib.Path(text)
    if not text or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Refused(refusal("CALCULATOR_NOT_CONFIGURED", f"{CALCULATOR_ENV} must name a blank copy of the USGBC "
                              "LEED v4.1 Daylight and Quality Views Calculator V02 (.xlsm), as an absolute path to a "
                              "plain file, in the client's server configuration."))
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _CHECKED:
        try:
            v02.check_template(Book.read(path))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError,
                RuntimeError) as exc:  # ValueError covers CalculatorUnsupported and UnicodeDecodeError
            raise Refused(refusal("CALCULATOR_UNSUPPORTED", f"{path.name}: {one_line(exc)}")) from None
        _CHECKED.add(key)
    return path


def check_calculator(folder: pathlib.Path, raw: dict, case: str) -> pathlib.Path:
    """The checks before a calculator job (calculator spec 4): the template,
    the units, and a VLT for every vision glazing when Quality Views ran."""
    template = calculator_template()
    section = raw.get("calculator") if isinstance(raw.get("calculator"), dict) else {}
    if section.get("units") not in ("IP", "SI"):
        raise Refused(refusal("FILL_ME_LEFT", "project.yaml has no calculator.units: ask the user whether the "
                              "project is registered in IP or SI units and set it (inputs action='update').",
                              field="calculator.units"))
    if result_path(folder, "views", case).is_file():
        roles = load_materials(folder / "materials.csv")
        have = vlt_values(folder / "materials.csv")
        missing = [m for m, role in roles.items() if role in VLT_ROLES and m not in have]
        if missing:
            raise Refused(refusal("INPUT_INVALID", f"materials.csv: {', '.join(missing)} need a vlt (the glass's "
                                  "visible light transmittance in percent, from the product data sheet) for the "
                                  "calculator's Quality Views rows; ask the user and set it (inputs "
                                  "action='update', file='materials').", field="file"))
    return template
