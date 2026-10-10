# leed_daylight/deliver/package.py
"""The submission folder (reports spec 8).

Built in ``submission/.<case>.building/`` and renamed at the end, so a
stopped or failed build never leaves half a package and never damages the
previous one. The prepared model is written again for this case with
:func:`~leed_daylight.deliver.data.prepare_case_model` (no Radiance),
because ``_run`` holds whichever case ran last, and (pre-flight ruling F15)
because a views-only case must be prepared exactly as ``drivers.views``
prepared it -- ``prepare_case_model`` already carries that distinction, so
this module never re-derives the scene -> materials -> rooms -> prepare
chain that ``deliver/data.py`` already owns (pre-flight ruling F11: one
prepare chain, not two).

The checksum and provenance pattern is ported from the author's earlier
work (PORTED.md).

Fix round 1 (review): the swap at the end of :func:`build_package` can be
interrupted (killed, or a failed rename) between renaming the old package
to ``.<case>.old`` and renaming the new one into place; ``.<case>.old`` is
therefore never deleted on sight -- it is restored to ``<case>/`` whenever
``<case>/`` is missing, and only ever discarded once a complete package
already sits at ``<case>/``. Every file this module copies into the
package is checked first: a symlinked source, or one that resolves outside
the project folder, is refused by name (``credit_rules_path`` is the one
source allowed to live outside the project folder, but not as a symlink).

Atlas spec 11: the package also holds the floor atlas the report step built
(``atlas/``: exactly the files the manifest's ``atlas`` entry lists, so
never ``blender.log``). It refuses, naming the report command, when the
entry is missing (reports made before the atlas was added have none, so an
existing project runs ``report`` once more), when the entry was built from
other input files, when it shows other options than the package holds, or
when an option's results file or ``points.csv`` changed after it was built.

Merge-split Review Focus 5: ``config/`` also holds ``splits.csv`` when the
project has one, so a reader can rebuild the same rooms; the README's
reproduce step then names it. Without the file the package is unchanged.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import sys

import yaml

from ..constants import TOOL_VERSION
from ..engine.jobs import checked_dir, refuse_link, replace_file
from ..engine.in_use import explain_in_use, remove_tree_explained
from ..project.inputs import presentation_digest, sha256_file, stale_inputs
from ..project.splits import SPLITS_FILE
from ..scene.input_file import read_input_text
from .atlas import BLEND_FILE, PAGE_FILE, atlas_dir
from .calculator.build import RECORD as CALCULATOR_RECORD, RECORD_FORMAT, calculator_dir, vlt_digest
from .data import CaseData, DeliverError, prepare_case_model, read_json_object, result_path
from .manifest import ATLAS_KEY, REPORT_MANIFEST, read_report_manifest

CHECKSUMS = "checksums.sha256"
#: The table of every package, at the submission folder's top.
INDEX = "README.md"
_LIBRARIES = ("numpy", "PyYAML", "shapely", "python-docx", "matplotlib", "Pillow", "cairosvg")
_LOGS = {"option1": ("radiance.log",), "option2": ("option2", "radiance.log"), "views": ("views", "radiance.log")}
_LOGS_NOTE = (
    "logs/<option>_radiance.log is the most recent Radiance run log for that option; "
    "_run/ keeps only the last run, so it may be from a different case than this package."
)


class PackageError(DeliverError):
    """The submission folder cannot be built from these reports."""


def _copy_checked(src: pathlib.Path, dest: pathlib.Path, base: pathlib.Path, *, allow_outside: bool = False) -> None:
    """Copy ``src`` to ``dest``, refusing by name a source that is a
    symlink, or (unless ``allow_outside``) one that resolves outside
    ``base`` -- the project folder. Both the report folder and the
    project folder are copied from largely as given, so a planted symlink
    must never be followed wherever it points (fix round 1, risk c).
    ``credit_rules_path`` is the one source allowed to live outside the
    project folder (the project config says so), but it is still refused
    if it is itself a symlink.
    """
    src = pathlib.Path(src)
    if src.is_symlink():
        raise PackageError(f"{src}: refusing to package a symlink")
    if not src.is_file():
        raise PackageError(f"{src}: not found")
    if not allow_outside:
        root = pathlib.Path(base).resolve()
        try:
            src.resolve().relative_to(root)
        except ValueError:
            raise PackageError(f"{src}: resolves outside the project folder {base}") from None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _check_reports(case: CaseData, reports: pathlib.Path) -> dict:
    manifest_path = reports / REPORT_MANIFEST
    if not manifest_path.is_file():
        raise PackageError(f"no reports for case {case.case!r}; run python -m leed_daylight.drivers.report first")
    manifest = read_report_manifest(manifest_path, error=PackageError)
    presentation = presentation_digest(case.project, case.folder)
    for option, data in case.options.items():
        entry = manifest["options"].get(option)
        if entry is None or entry.get("results_sha256") != sha256_file(data.results_path):
            raise PackageError(
                f"the {option} report for case {case.case!r} is missing or older than its results; "
                "run python -m leed_daylight.drivers.report again"
            )
        if entry.get("presentation_sha256") != presentation:
            raise PackageError(
                f"the {option} report for case {case.case!r} was made before the report: or atlas: section "
                "of project.yaml (or a drawing the atlas uses) last changed, or by an older version that "
                "did not record them; run python -m leed_daylight.drivers.report again"
            )
        points_path = data.results_path.parent / "points.csv"
        if entry.get("points_sha256") != sha256_file(points_path):
            raise PackageError(
                f"the {option} report for case {case.case!r} is missing or older than its points.csv; "
                "run python -m leed_daylight.drivers.report again"
            )
        files = entry.get("files")
        if not isinstance(files, list):
            raise PackageError(
                f"the {option} report manifest for case {case.case!r} has no 'files' list; "
                "run python -m leed_daylight.drivers.report again"
            )
        missing = [f for f in files if not (reports / f).is_file()]
        if missing:
            raise PackageError(
                f"report file {missing[0]} for {option} is missing; "
                "run python -m leed_daylight.drivers.report again"
            )
        if option == "option1":
            # Pre-flight ruling F6: glare_narrative.md is shared by every
            # case in the project, so the report that rendered it may be
            # stale (the file has changed since) or may belong to another
            # case entirely; either way the package must refuse by name.
            narrative_path = case.folder / "glare_narrative.md"
            stale = not narrative_path.is_file() or entry.get("narrative_sha256") != sha256_file(narrative_path)
            wrong_case = entry.get("narrative_case") != case.case
            if stale or wrong_case:
                raise PackageError(
                    f"glare_narrative.md has changed since the option1 report for case {case.case!r} was made "
                    "(or that report was made for a different case); "
                    "run python -m leed_daylight.drivers.report again"
                )
    _check_atlas(case, manifest, presentation)
    return manifest


_RERUN_REPORT = "run python -m leed_daylight.drivers.report again"


def _check_atlas(case: CaseData, manifest: dict, presentation: str) -> None:
    """Atlas spec 11: the manifest's ``atlas`` entry is present, was built
    from the current input files, shows exactly the options being packaged,
    records the current results file and ``points.csv`` of each, and every
    file it lists is on disk."""
    entry = manifest.get(ATLAS_KEY)
    if entry is None:
        raise PackageError(
            f"the reports for case {case.case!r} have no floor atlas (reports made before the atlas was "
            f"added have none); {_RERUN_REPORT}"
        )
    changed = stale_inputs(entry["inputs"], case.project, case.folder)
    if changed:
        raise PackageError(
            f"the floor atlas for case {case.case!r} was built from other inputs ({changed[0]} changed since); "
            f"{_RERUN_REPORT}"
        )
    if entry.get("presentation_sha256") != presentation:
        raise PackageError(
            f"the floor atlas for case {case.case!r} was made before the report: or atlas: section of "
            "project.yaml (or a drawing it uses) last changed, or by an older version that did not record "
            f"them; {_RERUN_REPORT}"
        )
    if set(entry["options"]) != set(case.options):
        shown = ", ".join(sorted(entry["options"])) or "no option"
        packaged = ", ".join(sorted(case.options))
        raise PackageError(
            f"the floor atlas for case {case.case!r} shows {shown} but the package holds {packaged}; "
            f"{_RERUN_REPORT}"
        )
    for option, data in case.options.items():
        recorded = entry["options"][option]
        for key, path, name in (("results_sha256", data.results_path, "results file"),
                                ("points_sha256", data.results_path.parent / "points.csv", "points.csv")):
            if recorded[key] != sha256_file(path):
                raise PackageError(
                    f"the floor atlas for case {case.case!r} is older than the {option} {name}; {_RERUN_REPORT}"
                )
    source = atlas_dir(case.folder, case.case)
    missing = [f for f in entry["files"] if not (source / f).is_file()]
    if missing:
        raise PackageError(f"atlas file {missing[0]} for case {case.case!r} is missing; {_RERUN_REPORT}")


def _summary(case: CaseData) -> dict:
    summary: dict = {"area_m2": None}
    if "option1" in case.options:
        r = case.options["option1"].results
        summary["option1"] = {"sda_area_pct": r["sda_area_pct"], "ase_area_pct": r["ase_area_pct"],
                              "points": r["score"]["points"]}
        summary["area_m2"] = sum(float(s["area_m2"]) for s in r["spaces"])
    if "option2" in case.options:
        r = case.options["option2"].results
        summary["option2"] = {"area_pct": r["area_pct"], "points": r["score"]["points"]}
        summary["area_m2"] = summary["area_m2"] or sum(float(s["area_m2"]) for s in r["spaces"])
    if "views" in case.options:
        r = case.options["views"].results
        summary["views"] = {"quality_area_pct": r["quality_area_pct"], "points": r["score"]["points"]}
        summary["area_m2"] = summary["area_m2"] or float(r["total_area_m2"])
    return summary


def _library_versions() -> dict:
    versions = {}
    for name in _LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _current_calculator(case: CaseData) -> tuple[pathlib.Path | None, bool]:
    """(the case's filled calculator, False) when its record's results
    hashes equal the results files now and its units and VLT values equal
    the inputs now (calculator spec D7) and it was filled in the current
    record format; (None, True) when a calculator was filled but is not
    current; (None, False) when none was."""
    folder = calculator_dir(case.folder, case.case)
    record_path = folder / CALCULATOR_RECORD
    if not record_path.is_file():
        return None, False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["format"] != RECORD_FORMAT:
            return None, True
        hashes = record["results"]
        path = folder / record["file"]
        if set(hashes) != set(case.options) or path.name != record["file"] or not path.is_file():
            return None, True
        if record["units"] != case.project.calculator.units:
            return None, True
        if record["vlt_sha256"] != vlt_digest(case.folder / "materials.csv"):
            return None, True
        for option, digest in hashes.items():
            current = hashlib.sha256(result_path(case.folder, option, case.case).read_bytes()).hexdigest()
            if current != digest:
                return None, True
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        return None, True
    return path, False


#: The README line when a filled calculator is left out of the package.
CALCULATOR_STALE = ("calculator: the filled calculator is older than the results or inputs; "
                    "run deliver(action='calculator') again")


def _points(value) -> str:
    return "not awarded" if value is None else str(value)


def _fmt(value, unit: str = " %") -> str:
    return "" if value is None else f"{float(value):.1f}{unit}"


def _atlas_rows(files) -> str:
    """The README rows of the atlas files in the package."""
    rows = [f"| `atlas/{PAGE_FILE}` | The floor atlas: one page per storey and a 3D view with every result "
            "layer; open it in a browser, no network needed |"]
    if BLEND_FILE in files:
        rows.append(f"| `atlas/{BLEND_FILE}` | The same model and result layers for Blender 4.2 or newer |")
    return "".join(row + "\n" for row in rows)


def _calculator_row(name: str | None) -> str:
    if name is None:
        return ""
    return (f"| `calculator/{name}` | The USGBC Daylight and Quality Views Calculator, filled from this case; "
            "open it in Excel, which recalculates it |\n")


def _readme(case: CaseData, summary: dict, config: pathlib.Path, atlas_files=(), calculator: str | None = None,
            calculator_stale: bool = False) -> str:
    rows = ["| Option | Result | Points |", "|---|---:|---:|"]
    if "option1" in summary:
        s = summary["option1"]
        rows.append(f"| Daylight Option 1 | sDA {_fmt(s['sda_area_pct'])}, ASE {_fmt(s['ase_area_pct'])} | {_points(s['points'])} |")
    if "option2" in summary:
        rows.append(f"| Daylight Option 2 | {_fmt(summary['option2']['area_pct'])} of floor area in the band | {_points(summary['option2']['points'])} |")
    if "views" in summary:
        rows.append(f"| Quality Views | {_fmt(summary['views']['quality_area_pct'])} with a Quality View | {_points(summary['views']['points'])} |")
    skipped = "".join(f"- {o}: {reason}\n" for o, reason in case.skipped.items())
    skipped = (skipped + (f"- {CALCULATOR_STALE}\n" if calculator_stale else "")) or "- none\n"
    return (
        f"# {case.project.name}, case {case.case}\n\n"
        f"Built {case.created.isoformat()} by leed-daylight {TOOL_VERSION}.\n\n"
        "## Results\n\n" + "\n".join(rows) + "\n\n"
        "## Start here\n\n"
        "| File | What it is |\n|---|---|\n"
        "| `reports/<option>/00_PROJECT_<option>.docx` | The project report of each option |\n"
        "| `reports/<option>/<storey>_<option>.docx` | One report per storey |\n"
        "| `reports/option1/glare_narrative.md` | The glare narrative as reported; to change it, edit "
        "`<project folder>/glare_narrative.md`, then run the report and the package again |\n"
        "| `results/<option>/` | The results file and the per-point data |\n"
        + _atlas_rows(atlas_files) + _calculator_row(calculator) +
        f"| `logs/<option>_radiance.log` | {_LOGS_NOTE} |\n"
        "| `provenance.json` | Versions, input fingerprints and the results summary |\n\n"
        "## Options left out\n\n" + skipped + "\n"
        "## Check this folder\n\n"
        f"    shasum -a 256 -c {CHECKSUMS}\n\n"
        "## Reproduce\n\n" + _reproduce(case, config)
    )


#: The verb that computes each option's results, in run order (CLI spec 7.3).
_RUN_VERBS = (("option1", "run"), ("option2", "option2"), ("views", "views"))


def _reproduce(case: CaseData, config: pathlib.Path) -> str:
    """CLI spec 7.3: where each file goes, read from the copied
    ``config/project.yaml`` as written, and the ``leed-daylight`` commands
    for the options that ran.

    I1: when option1 ran and its glare narrative was copied into the
    package (:func:`_fill`, only when ``<folder>/glare_narrative.md``
    existed), the narrative in the package is the person's edited text --
    the plain ``report`` command below would instead regenerate a fresh
    draft, so a step must send it back to ``<folder>/glare_narrative.md``
    before ``report`` runs.
    """
    raw = yaml.safe_load(read_input_text(config / "project.yaml", encoding="utf-8"))
    rules = raw["standard"]["credit_rules"]
    scenes = ", ".join(f"`{name}`" for name in raw["scene"]["files"])
    weather = raw["weather"]["file"]
    verbs = [verb for option, verb in _RUN_VERBS if option in case.options] + ["report", "package"]
    commands = "".join(f'    leed-daylight {verb} "<folder>" --case "{case.case}"\n' for verb in verbs)
    copied = ["project.yaml", "materials.csv", "spaces.csv"]
    if (config / SPLITS_FILE).is_file():
        copied.append(SPLITS_FILE)
    points = (raw.get("grid") or {}).get("points_file")
    names = [f"`config/{name}`" for name in copied]
    steps = [
        f"Copy {', '.join(names[:-1])} and {names[-1]} into it.",
        f"Put `config/credit_rules.yaml` at `{rules}` (the path `standard.credit_rules` names).",
        f"Put the scene files at {scenes} (the paths `scene.files` names).",
        f"Put the weather file at `{weather}` (the path `weather.file` names); its sha256 is in "
        "`provenance.json`.",
    ]
    if points is not None:
        steps.append(f"Put `config/{pathlib.PurePath(points).name}` at `{points}` (the path `grid.points_file` names).")
    if "option1" in case.options and (case.folder / "glare_narrative.md").is_file():
        steps.append(
            "Copy `reports/option1/glare_narrative.md` to `glare_narrative.md` in the folder before you "
            "run `report`."
        )
    steps.append("Run, with the folder's path in place of `<folder>`:\n\n" + commands)
    numbered = "".join(f"{i}. {step}\n" for i, step in enumerate(steps, start=1))
    return (
        "Make a new folder, then:\n\n" + numbered +
        "Without the installed `leed-daylight` script, `python -m leed_daylight` takes the same verbs.\n"
    )


def _write_checksums(root: pathlib.Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix())
        if path.is_file() and not path.is_symlink() and path.name != CHECKSUMS
    ]
    (root / CHECKSUMS).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_checksums(folder: pathlib.Path) -> list[str]:
    folder = pathlib.Path(folder)
    root = folder.resolve()
    bad = []
    text = (folder / CHECKSUMS).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise PackageError(f"{CHECKSUMS}: malformed line {line!r}")
        digest, name = parts
        path = folder / name
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise PackageError(f"{CHECKSUMS}: {name!r} resolves outside {folder}") from None
        if not path.is_file() or sha256_file(path) != digest:
            bad.append(name)
    return bad


def _confined(root: pathlib.Path, rel: str) -> pathlib.Path:
    """``root / rel``, refused by name when it resolves outside ``root``
    (final review I5/T12: a manifest entry never writes outside the package
    being built, whatever the manifest reader let through)."""
    dest = pathlib.Path(root) / rel
    try:
        dest.resolve().relative_to(pathlib.Path(root).resolve())
    except ValueError:
        raise PackageError(f"{REPORT_MANIFEST}: {rel!r} resolves outside {root}") from None
    return dest


def _fill(case: CaseData, building: pathlib.Path, now: datetime.datetime) -> None:
    folder, project = case.folder, case.project
    reports = folder / "_run" / "reports" / case.case
    manifest = _check_reports(case, reports)
    for option in case.options:
        # Fix round 1, risk c: copy exactly the files the report manifest
        # lists for this option, not the whole folder -- and refuse (by
        # name) a symlink or a path that resolves outside the project.
        for rel in manifest["options"][option]["files"]:
            _copy_checked(_confined(reports, rel), _confined(building / "reports", rel), folder)
        results = building / "results" / option
        results.mkdir(parents=True)
        source = case.options[option].results_path
        _copy_checked(source, results / source.name, folder)
        _copy_checked(source.parent / "points.csv", results / "points.csv", folder)
        log = folder.joinpath("_run", *_LOGS[option])
        if log.is_file():
            (building / "logs").mkdir(exist_ok=True)
            _copy_checked(log, building / "logs" / f"{option}_radiance.log", folder)
    # Atlas spec 11: exactly the files the atlas entry lists (never
    # blender.log), with the same symlink and confinement checks.
    atlas_files = manifest[ATLAS_KEY]["files"]
    source = atlas_dir(folder, case.case)
    for rel in atlas_files:
        _copy_checked(_confined(source, rel), _confined(building / "atlas", rel), folder)
    calculator, calculator_stale = _current_calculator(case)
    if calculator is not None:
        _copy_checked(calculator, building / "calculator" / calculator.name, folder)
    if "option1" in case.options and (folder / "glare_narrative.md").is_file():
        _copy_checked(
            folder / "glare_narrative.md", building / "reports" / "option1" / "glare_narrative.md", folder,
        )

    config = building / "config"
    config.mkdir()
    for name in ("project.yaml", "materials.csv", "spaces.csv"):
        _copy_checked(folder / name, config / name, folder)
    if (folder / SPLITS_FILE).exists() or (folder / SPLITS_FILE).is_symlink():
        _copy_checked(folder / SPLITS_FILE, config / SPLITS_FILE, folder)
    if project.grid.points_file is not None:
        # Like the credit rules, the points file may live outside the project folder; it may be no link.
        taken = {"project.yaml", "materials.csv", "spaces.csv", "credit_rules.yaml", SPLITS_FILE}
        # Compared without case: on macOS and Windows `Spaces.csv` would overwrite `spaces.csv`.
        if project.grid.points_file.name.casefold() in {n.casefold() for n in taken}:
            raise PackageError(
                f"grid.points_file is named {project.grid.points_file.name!r}, which is also a file the package "
                "writes to config/; rename the points file"
            )
        _copy_checked(project.grid.points_file, config / project.grid.points_file.name, folder, allow_outside=True)
    # credit_rules_path may legitimately live outside the project folder
    # (the project config allows it); it must still not be a symlink.
    _copy_checked(project.credit_rules_path, config / "credit_rules.yaml", folder, allow_outside=True)

    # F11/F15: the one scene -> prepare chain, shared with ``load_case``;
    # it already knows how to prepare a views-only case (module-global
    # name, so a test can monkeypatch a failure while writing the model).
    prepare_case_model(project, case.spaces, case.case, building / "model")

    summary = _summary(case)
    first = next(iter(case.options.values()))
    provenance = {
        "case": case.case, "created": now.isoformat(timespec="seconds"), "tool_version": TOOL_VERSION,
        "radiance_version": {o: d.results.get("radiance_version") for o, d in case.options.items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": _library_versions(), "inputs": dict(first.results["inputs"]),
        "options_included": list(case.options), "options_skipped": dict(case.skipped), "summary": summary,
        "atlas_files": list(atlas_files), "notes": [_LOGS_NOTE],
    }
    (building / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        newline="\n",
    )
    (building / "README.md").write_text(_readme(case, summary, config, atlas_files, calculator.name if calculator else None,
                                                       calculator_stale), encoding="utf-8", newline="\n")
    _write_checksums(building)


def _check_inside(folder: pathlib.Path, submission_root: pathlib.Path, names) -> None:
    """Refuse (``UnsafePath``), by name, a link at ``submission_root`` or
    above it in the project folder, or at any of ``names`` in it, before
    anything is deleted, renamed or created (re-review OOS-1): a link there
    would send the deletes and renames of :func:`build_package` outside the
    project folder."""
    try:
        parts = submission_root.relative_to(folder).parts
    except ValueError:
        raise PackageError(f"{submission_root}: is not inside the project folder {folder}") from None
    for name in names:
        checked_dir(folder, *parts, name, make=False)


def build_package(
    case: CaseData, *, submission_root: pathlib.Path, now: datetime.datetime, out=None,
) -> pathlib.Path:
    out = sys.stdout if out is None else out
    submission_root = pathlib.Path(submission_root)
    final = submission_root / case.case
    building = submission_root / f".{case.case}.building"
    old = submission_root / f".{case.case}.old"
    _check_inside(pathlib.Path(case.folder), submission_root, (final.name, building.name, old.name))
    refuse_link(submission_root / INDEX)
    submission_root.mkdir(parents=True, exist_ok=True)

    # Fix round 1 (IMPORTANT): a previous run may have been killed, or
    # failed, mid-swap -- after `final.rename(old)` but before
    # `building.rename(final)` completed. `.old` then holds the only
    # complete package on disk, with `final` missing; it must be restored,
    # never discarded, whenever `final` is missing. Only once `final`
    # already exists again (the swap did complete) is a leftover `.old`
    # a spent copy, safe to drop. `.building` never held a promoted
    # package, so a leftover one is always safe to discard outright.
    if building.exists():
        remove_tree_explained(building)
    if old.exists():
        if final.exists():
            remove_tree_explained(old)
        else:
            old.rename(final)

    building.mkdir()
    try:
        _fill(case, building, now)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    swapped_out = final.exists()
    if swapped_out:
        with explain_in_use(final, folder=True):
            final.rename(old)
    try:
        building.rename(final)
    except BaseException:
        # The new package could not be swapped into place: restore the
        # previous one (if there was one) and leave no `.building`/`.old`
        # behind, so a package always exists if one did before this call.
        if swapped_out:
            old.rename(final)
        shutil.rmtree(building, ignore_errors=True)
        raise
    if old.exists():
        remove_tree_explained(old)
    write_index(submission_root, out=out, root=case.folder)
    return final


def write_index(submission_root: pathlib.Path, *, out=None, root=None) -> pathlib.Path:
    """Write ``<submission_root>/README.md``, the table of every package,
    never through a link: at it, or at a folder from ``root`` (default:
    ``submission_root`` itself) down to it (security review H1's class)."""
    out = sys.stdout if out is None else out
    rows = ["| Case | Area m2 | Option 1 sDA | Option 1 ASE | Option 1 points | Option 2 in band | "
            "Option 2 points | Quality View | Views points |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for folder in sorted(p for p in pathlib.Path(submission_root).iterdir()
                         if p.is_dir() and not p.name.startswith(".")):
        provenance_path = folder / "provenance.json"
        if not provenance_path.is_file():
            continue
        try:
            s = read_json_object(provenance_path).get("summary")
            if not isinstance(s, dict):
                raise TypeError(f"'summary' is a {type(s).__name__}, not an object")
            o1, o2, v = (s.get(k, {}) for k in ("option1", "option2", "views"))
            for key, value in (("option1", o1), ("option2", o2), ("views", v)):
                if not isinstance(value, dict):
                    raise TypeError(f"'summary.{key}' is a {type(value).__name__}, not an object")
            row = (
                f"| {folder.name} | {_fmt(s.get('area_m2'), '')} | {_fmt(o1.get('sda_area_pct'))} | "
                f"{_fmt(o1.get('ase_area_pct'))} | {o1.get('points', '')} | {_fmt(o2.get('area_pct'))} | "
                f"{o2.get('points', '')} | {_fmt(v.get('quality_area_pct'))} | {v.get('points', '')} |"
            )
        except (DeliverError, TypeError, ValueError) as exc:
            # Fix round 1: this case's own report/package already
            # succeeded and has been swapped into place by the time the
            # index is written; another, unrelated case's malformed
            # provenance.json (not UTF-8, not JSON, or an odd shape -- final
            # review I5) must not fail the whole command -- skip its row and
            # say so, rather than raise after the swap.
            print(f"warning: skipping {folder.name!r} in the submission index: "
                  f"{provenance_path} is malformed ({exc})", file=out)
            continue
        rows.append(row)
    path = pathlib.Path(submission_root) / INDEX
    text = "# Submission packages\n\n" + "\n".join(rows) + "\n"
    replace_file(submission_root if root is None else root, path, text.encode("utf-8"))
    return path
