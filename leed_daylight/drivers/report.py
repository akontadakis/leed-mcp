# leed_daylight/drivers/report.py
"""The ``report`` command (reports spec 9): Word reports, plots and the glare
narrative draft for one case.

    python -m leed_daylight.drivers.report "<project folder>" [--case <name>] [--option option1|option2|views] [--renderer matplotlib|cairosvg|pillow]

It writes ``<folder>/_run/reports/<case>/<option>/`` for every option whose
results exist and are current, and names every option it skipped. Option 1
also refreshes ``<folder>/glare_narrative.md`` and renders it to Word.

Pre-flight ruling F1: ``out`` defaults to ``None`` and is resolved to
``sys.stdout`` *inside* this function, not at import time. An import-time
default (``out=sys.stdout`` in the signature) is bound once, when this
module is first imported -- before a test's ``capsys`` (or any other stdout
replacement) has taken effect -- so prints would silently go to the real
stdout and never reach the test.

Pre-flight ruling F3: the report manifest is *merged*, not rewritten from
scratch. Entries for options this run does not touch (neither built nor
skipped, because they are not in ``options``) are kept exactly as they were
read from the manifest already on disk, so running ``--option views`` after
a full report does not erase the option1/option2 entries Task 12's package
step reads.

Pre-flight ruling F4: the python-docx-dependent modules (``deliver.word.*``,
``deliver.narrative``) are imported inside :func:`report_folder`, in a
``try`` -- never at this module's top -- so importing this module never
requires python-docx, and a missing ``python-docx`` at report time is a
named :class:`ReportError`, never a traceback. Controller ruling (fix round
1): that import happens after the folder is checked and the case is loaded,
so an invalid folder or a no-usable-results refusal is reported before (and
instead of) a missing-python-docx refusal, not after.

Pre-flight ruling F6: one ``glare_narrative.md`` is shared by every case in
a project (spec 7). The manifest records its sha256 and the case it was
rendered for on the ``option1`` entry, so a later package step (Task 12) can
refuse to package a report whose narrative has since changed on disk.

Pre-flight ruling F16: every file this run writes -- each report ``.docx``
and each plot ``.png`` -- is printed, in the order it is written.

Fix round 1 (Task 12 review): each option's manifest entry also records
``points_sha256`` (its ``points.csv``'s own sha256) beside
``results_sha256``, so the package step can refuse a report whose
``points.csv`` changed after the report was made even though the results
file itself did not.

Atlas spec 11: after the Word files, the command builds the floor atlas of
the same case (``deliver.atlas.build_atlas``, always with every option, so
``--option views`` still gives an atlas of every current result) and puts
the entry it returns into the manifest as ``atlas``. ``--no-blender`` is
passed through. A Blender failure is a warning (exit 0). Any other atlas
failure is raised as it is (``main`` prints ``error: ...``, exit 1) after
the manifest has been written with the Word entries and no ``atlas`` entry,
so ``package`` refuses until the atlas is built.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
from collections.abc import Sequence

import yaml
from shapely.errors import ShapelyError

from ..deliver.atlas import build_atlas
from ..deliver.data import OPTIONS, load_case
from ..deliver.figures.base import get_renderer
from ..deliver.manifest import (
    ATLAS_KEY,
    REPORT_MANIFEST,
    read_report_manifest,
    report_entry,
    write_report_manifest,
)
from ..engine.jobs import checked_dir, folder_lock, refuse_link
from ..engine.in_use import explain_in_use, remove_tree_explained
from ..project.config import RENDERERS
from ..project.inputs import presentation_digest
from ._folder import REPO_ROOT, check_folder


class ReportError(ValueError):
    """The report command cannot proceed from these inputs."""


def _word_modules():
    """Import the python-docx-dependent modules this command needs.

    Imported here, not at this module's top (pre-flight ruling F4): a
    missing ``python-docx`` then surfaces as a named :class:`ReportError`
    naming the package and the install command, instead of a
    ``ModuleNotFoundError`` traceback at ``import leed_daylight.drivers.report``.
    """
    try:
        from ..deliver import narrative
        from ..deliver.word import common
        from ..deliver.word import option1, option2, views
    except ImportError as exc:
        top = (exc.name or "").split(".")[0]
        if top != "docx":
            raise
        raise ReportError(
            f"the report command needs the Python package 'docx' ({exc}); "
            'install the report extras: pip install -e ".[report]"'
        ) from None
    builders = {"option1": option1, "option2": option2, "views": views}
    return narrative, common, builders


def _save(built, path: pathlib.Path, plots: pathlib.Path, written: list[pathlib.Path]) -> None:
    with explain_in_use(path):
        built.document.save(path)
    written.append(path)
    for name, png in built.figures:
        target = plots / name
        if not target.exists():
            target.write_bytes(png)
            written.append(target)


def _check_storey_names(data, safe_name) -> None:
    """Final review M7: two storey labels that become the same file name
    ("L 1" and "L_1") would overwrite one storey's report with the other's;
    refuse by name before anything is written. CLI spec 7.1: compared
    casefolded, because "L1" and "l1" are one file on a case-insensitive
    disk (the macOS default)."""
    seen: dict[str, str] = {}
    for storey in data.storeys:
        name = safe_name(storey.label)
        key = name.casefold()
        if key in seen:
            raise ReportError(
                f"spaces.csv: storeys {seen[key]!r} and {storey.label!r} both give the report file name "
                f"{name}_<option>.docx; rename one of them in the floor column"
            )
        seen[key] = storey.label


def _load_manifest(path: pathlib.Path) -> dict:
    """The manifest already on disk (F3: merged, not rewritten), or an empty
    one; a malformed one is refused naming the file (final review I5). Its
    ``atlas`` entry is left out unread: this run replaces it (atlas final
    review M1), so a bad one never blocks the command that fixes it."""
    if not path.is_file():
        return {"options": {}, "skipped": {}}
    return read_report_manifest(path, check_atlas=False)


def report_folder(folder, case: str | None = None, *, options: Sequence[str] = OPTIONS,
                  renderer: str | None = None, out=None,
                  today: datetime.date | None = None, blender: bool = True) -> pathlib.Path:
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    # Controller ruling (fix round 1): check the folder and load the case
    # before importing the python-docx-dependent modules, so an invalid
    # folder or a no-usable-results refusal is reported before (and instead
    # of) a missing-python-docx refusal.
    check_folder(folder, REPO_ROOT, ReportError)
    data = load_case(folder, case, options=options, today=today)
    narrative, common, builders = _word_modules()
    _check_storey_names(data, common.safe_name)
    painter = get_renderer(renderer or data.project.report.renderer)
    # Re-review OOS-1: a link at _run, _run/reports, the case folder or an
    # option folder (or _run/atlas, for the atlas built below) would send a
    # delete outside the project folder; refuse it, by name, before
    # anything is deleted or created.
    for option in options:
        checked_dir(folder, "_run", "reports", data.case, option, make=False)
    checked_dir(folder, "_run", "atlas", make=False)
    refuse_link(checked_dir(folder, make=False) / "glare_narrative.md")
    base = folder / "_run" / "reports" / data.case
    base.mkdir(parents=True, exist_ok=True)

    manifest_path = base / REPORT_MANIFEST
    manifest = _load_manifest(manifest_path)
    manifest["case"] = data.case
    manifest["created"] = data.created.isoformat()
    manifest["renderer"] = painter.name
    manifest_options = manifest["options"]
    manifest_skipped = manifest["skipped"]

    presentation = presentation_digest(data.project, folder)
    for option in options:
        target = base / option
        if target.exists():
            remove_tree_explained(target)
        if option not in data.options:
            reason = data.skipped[option]
            print(f"skipped {option}: {reason}", file=out)
            manifest_options.pop(option, None)
            manifest_skipped[option] = reason
            continue
        manifest_skipped.pop(option, None)
        plots = target / "plots"
        plots.mkdir(parents=True)
        written: list[pathlib.Path] = []
        builder = builders[option]
        _save(builder.build_project(data, painter), target / f"00_PROJECT_{option}.docx", plots, written)
        for storey in data.storeys_for(option):
            _save(builder.build_storey(data, storey.label, painter),
                  target / f"{common.safe_name(storey.label)}_{option}.docx", plots, written)

        if option == "option1":
            md_path = folder / "glare_narrative.md"
            for warning in narrative.update_narrative(md_path, data):
                print(f"warning: {warning}", file=out)
            md_text = md_path.read_text(encoding="utf-8")
            docx_path = target / "glare_narrative.docx"
            narrative.render_docx(md_text, docx_path, data)
            written.append(docx_path)
            print(f"{narrative.count_marks(md_text)} open items in glare_narrative.md ([ADD] and [CONFIRM])",
                  file=out)

        # Pre-flight ruling F6: the option1 entry also records the narrative,
        # so a later package step can tell whether it has changed since.
        entry = report_entry(
            data.options[option].results_path, (p.relative_to(base).as_posix() for p in written),
            narrative_path=md_path if option == "option1" else None, case=data.case,
            presentation=presentation,
        )
        manifest_options[option] = entry

        # Pre-flight ruling F16: every file written is printed, plots too.
        for path in written:
            print(f"wrote {path}", file=out)

    for warning in data.warnings:
        print(f"warning: {warning}", file=out)

    # Atlas spec 11: the atlas entry is set only by a build that succeeded;
    # the manifest is written either way, with the Word entries.
    manifest.pop(ATLAS_KEY, None)
    try:
        atlas = build_atlas(folder, data.case, blender=blender, data=data, out=out)
        manifest[ATLAS_KEY] = atlas.entry
    finally:
        write_report_manifest(manifest_path, manifest)
    # Atlas final review M4: the atlas holds every option, so it may leave
    # out one this run did not ask for; say why, once.
    reported = {option for option in options if option not in data.options}
    for option, reason in atlas.skipped.items():
        if option not in reported:
            print(f"skipped {option}: {reason}", file=out)
    for warning in atlas.warnings:
        if warning not in data.warnings:
            print(f"warning: {warning}", file=out)
    if atlas.blender_failed:
        print(f"warning: {atlas.blender}", file=out)
    elif atlas.blender:
        print(atlas.blender, file=out)
    return base


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "python -m leed_daylight.drivers.report",
        description="Write the Word reports, plots and glare narrative draft for one case.",
    )
    parser.add_argument("folder", help="the project folder")
    parser.add_argument("--case", default=None, help="the case (default: the first case in spaces.csv)")
    parser.add_argument("--option", choices=OPTIONS, default=None, help="only this option (default: all three)")
    parser.add_argument("--renderer", choices=RENDERERS, default=None,
                        help="figure renderer (default: report.renderer in project.yaml, else matplotlib)")
    parser.add_argument("--no-blender", action="store_true",
                        help="do not look for Blender or build the atlas's model.blend")
    args = parser.parse_args(argv)
    options = (args.option,) if args.option else OPTIONS
    try:
        check_folder(pathlib.Path(args.folder), REPO_ROOT, ReportError)
        with folder_lock(pathlib.Path(args.folder), "report"):
            report_folder(args.folder, args.case, options=options, renderer=args.renderer,
                          blender=not args.no_blender)
    except (ValueError, OSError, yaml.YAMLError, ShapelyError) as exc:
        text = " ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {text or type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
