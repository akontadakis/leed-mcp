# leed_daylight/deliver/atlas/__init__.py
"""The floor atlas and the Blender file of one case (atlas spec).

``build_atlas`` writes ``<folder>/_run/atlas/<case>/``:

* ``bundle.npz``: the shared data (``bundle.write_npz``);
* ``atlas.html``: the page (``page.write_page``);
* ``model.blend``: only when Blender is wanted and a Blender 4.2 or newer
  is found (``engine/blender.py`` runs ``blender_build.py``, which this
  package never imports);
* ``blender.log``: Blender's output, kept when Blender ran; never listed
  in the manifest entry (it holds machine paths).

The case is always loaded with every option and ``require_results=False``,
so the atlas shows every current result and, before any run, is a model
check. ``report`` passes the ``CaseData`` it loaded when that holds every
option. The model is prepared again, into a temporary folder, and handed
to ``on_prepared`` (the ``atlas`` verb prints the case summary from it).
Every refusal (the case, a drawing, the prepare chain) comes before the
folder is emptied and written. Blender's outcome never stops the page: not
found, too old or unreadable is one line; a failed run is a line and
``blender_failed``, and the caller decides what that means (spec 12).

``build_atlas`` returns the report-manifest entry and never writes the
manifest itself (spec 11).
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
import tempfile
import types
from collections.abc import Callable, Mapping

from ...build.prepare import PreparedModel
from ...engine.blender import (
    BlenderFailed,
    BlenderNotFound,
    BlenderRunner,
    BlenderTooOld,
    check_version,
    find_blender,
    run_blender,
    subprocess_blender_runner,
)
from ...engine.jobs import checked_dir
from ...engine.in_use import remove_tree_explained
from ...project.inputs import input_fingerprints, presentation_digest
from ..data import OPTIONS, CaseData, DeliverError, load_case, prepare_case_model
from ..manifest import atlas_entry
from .bundle import build_bundle, write_npz
from .drawing import load_drawings
from .page import size_warning, write_page

#: The script Blender runs; read by Blender, never imported here.
BLENDER_SCRIPT = pathlib.Path(__file__).resolve().parent / "blender_build.py"
PAGE_FILE = "atlas.html"
BUNDLE_FILE = "bundle.npz"
BLEND_FILE = "model.blend"
BLENDER_LOG = "blender.log"


class AtlasError(DeliverError):
    """The atlas cannot be built from these inputs."""


@dataclasses.dataclass(frozen=True)
class AtlasResult:
    """What ``build_atlas`` did: the folder, the files written (relative,
    in the order written; ``blender.log`` never among them), the result
    layers shown, the options not shown with their reasons, the Blender
    line (empty when Blender was not asked for), whether a Blender run
    failed, the manifest entry and the warnings to print."""
    folder: pathlib.Path
    files: tuple[str, ...]
    layers: tuple[str, ...]
    skipped: Mapping[str, str]
    blender: str
    blender_failed: bool
    entry: dict
    warnings: tuple[str, ...]


def atlas_dir(folder, case: str) -> pathlib.Path:
    """``<folder>/_run/atlas/<case>``."""
    return pathlib.Path(folder) / "_run" / "atlas" / case


def _holds_every_option(data: CaseData) -> bool:
    return set(OPTIONS) <= set(data.options) | set(data.skipped)


def _blender(target: pathlib.Path, runner: BlenderRunner | None) -> tuple[str, bool]:
    """(the Blender line, whether a run failed); ``model.blend`` is in
    ``target`` only when the line says it was built."""
    runner = subprocess_blender_runner() if runner is None else runner
    try:
        exe = find_blender()
        version = check_version(exe, runner)
    except (BlenderNotFound, BlenderTooOld) as exc:
        return f"Blender skipped, no {BLEND_FILE}: {exc}", False
    try:
        run_blender(exe, BLENDER_SCRIPT, target / BUNDLE_FILE, target / BLEND_FILE,
                    log_path=target / BLENDER_LOG, runner=runner)
    except BlenderFailed as exc:
        return f"Blender failed, no {BLEND_FILE}: {exc}", True
    return f"{BLEND_FILE} built by Blender {'.'.join(str(v) for v in version)} ({exe})", False


def build_atlas(folder, case: str | None = None, *, blender: bool = True, data: CaseData | None = None,
                on_prepared: Callable[[PreparedModel], None] | None = None,
                blender_runner: BlenderRunner | None = None, out=None) -> AtlasResult:
    """Write the atlas of ``case`` (default: the first case) and return
    what was done. Prints ``wrote <path>`` for every file, in order, to
    ``out`` (default ``sys.stdout``, resolved here)."""
    out = sys.stdout if out is None else out
    folder = pathlib.Path(folder)
    if data is not None and case is not None and case != data.case:
        raise AtlasError(f"the loaded case is {data.case!r}, not {case!r}")
    if data is None or not _holds_every_option(data):
        data = load_case(folder, case if data is None else data.case, require_results=False)
    drawings = load_drawings(data)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = prepare_case_model(data.project, data.spaces, data.case, pathlib.Path(tmp)).prepared
        if on_prepared is not None:
            on_prepared(prepared)
        bundle = build_bundle(data, prepared)

    # Re-review OOS-1: a link at _run or _run/atlas would send the delete
    # below outside the project folder; refuse it, by name, first.
    checked_dir(folder, "_run", "atlas", make=False)
    target = atlas_dir(folder, data.case)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        remove_tree_explained(target)
    target.mkdir(parents=True)

    files: list[str] = []

    def wrote(name: str) -> None:
        files.append(name)
        print(f"wrote {target / name}", file=out)

    write_npz(bundle, target / BUNDLE_FILE)
    wrote(BUNDLE_FILE)
    n_bytes = write_page(target / PAGE_FILE, data, bundle, drawings)
    wrote(PAGE_FILE)
    warnings = list(data.warnings)
    too_big = size_warning(n_bytes)
    if too_big:
        warnings.append(too_big)

    line, failed = "", False
    if blender:
        line, failed = _blender(target, blender_runner)
        if (target / BLEND_FILE).is_file():
            wrote(BLEND_FILE)

    entry = atlas_entry(files, input_fingerprints(data.project, folder),
                        {option: loaded.results_path for option, loaded in data.options.items()},
                        presentation_digest(data.project, folder))
    return AtlasResult(
        folder=target, files=tuple(files), layers=tuple(layer.key for layer, _ in bundle.layers),
        skipped=types.MappingProxyType(dict(data.skipped)), blender=line, blender_failed=failed,
        entry=entry, warnings=tuple(warnings),
    )
