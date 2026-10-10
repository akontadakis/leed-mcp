# leed_daylight/project/inputs.py
"""Fingerprints of the files a result was computed from (reports spec 3.2).

Every results file records ``inputs``: a sha256 per input file. The report
and package steps compare them with the files on disk, so a report is never
built from results that no longer match the project. The prepared model's
``manifest.json`` is deliberately not one of them: every run of any case or
option rewrites it, so it would flag valid results as stale.

``project.yaml`` is the one input not hashed as bytes (final review I1,
Ruling FR-I1). Its top-level ``report:`` section (deviations, prepared_by,
the figure renderer) is presentation only and changes no result, so adding a
deviation must not force hours of Radiance reruns; so is its ``atlas:``
section (the drawings under the atlas plans, atlas spec 8.1); so is its
``calculator:`` section (the calculator's unit system changes no calculation,
only how the workbook is filled).
``project.yaml`` is therefore fingerprinted as the sha256 of a canonical
dump (``sort_keys``) of its parsed content with the ``report``, ``atlas`` and
``calculator`` keys removed, still keyed ``project.yaml``. The cost: an edit that changes only comments, whitespace,
key order or quoting in ``project.yaml`` no longer marks results stale
either, since none of it changes a result. Every other file is hashed as
bytes.

``splits.csv`` (merge-split spec 8) is optional and fingerprinted only when
it exists, so a project without it keeps the fingerprints it had. A result
recorded without the key is stale once the file exists, and one recorded
with it is stale when the file changes or is removed: a key on one side
only is already stale. The sensor points file (``grid.points_file``) is
fingerprinted only when ``project.yaml`` names it, for the same reason.

``sha256_file`` is the one chunked file hasher in this codebase (F11): it is
public so ``deliver/package.py`` and ``drivers/report.py`` reuse it instead
of each writing their own; it hashes outputs too, so it has no size cap.
``input_fingerprints`` hashes each input with ``_sha256_input`` instead,
which opens it as every input file is opened (``scene.input_file.open_input``):
a FIFO or device is refused at once, never waited on, and a file over
``INPUT_FILE_MAX`` bytes is refused (``InputFileError``).
"""
from __future__ import annotations

import csv
import hashlib
import io
import pathlib
from collections.abc import Mapping

import yaml

from ..scene.input_file import open_input, read_input_bytes, read_input_text, too_large
from .config import Project
from .splits import SPLITS_FILE

_CHUNK = 1 << 20
#: Top-level project.yaml sections that change no result (see above).
_PRESENTATION_SECTIONS = frozenset({"report", "atlas", "calculator"})
#: The presentation sections the Word reports and the atlas are made from.
_REPORT_SECTIONS = ("report", "atlas")


def sha256_file(path: str | pathlib.Path) -> str:
    """The sha256 hex digest of the file at ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_input(path: str | pathlib.Path) -> str:
    """``sha256_file`` of an input file, opened as every input file is
    (``scene.input_file.open_input``): only a plain file, never waited on,
    and at most ``INPUT_FILE_MAX`` bytes, or an ``InputFileError``. The same
    digest as ``sha256_file`` for any file it reads."""
    digest = hashlib.sha256()
    handle, limit = open_input(path)
    read = 0
    with handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            read += len(chunk)
            if read > limit:
                raise too_large(path, limit)
            digest.update(chunk)
    return digest.hexdigest()


def _project_yaml_digest(path: str | pathlib.Path) -> str:
    """sha256 of ``project.yaml``'s parsed content without its top-level
    ``report``, ``atlas`` and ``calculator`` keys, dumped canonically (sorted keys) -- see
    the module docstring (I1)."""
    raw = yaml.safe_load(read_input_text(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if k not in _PRESENTATION_SECTIONS}
    canonical = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def presentation_digest(project: Project, folder: pathlib.Path) -> str:
    """sha256 of what the Word reports and the atlas show beyond the results:
    the ``report:`` and ``atlas:`` sections of project.yaml (canonical, so
    key order and quoting do not count) and the bytes of each drawing the
    atlas names. The report step records it and the package step compares
    it, so an edit to either section after the reports were built is seen."""
    folder = pathlib.Path(folder)
    raw = yaml.safe_load(read_input_text(folder / "project.yaml", encoding="utf-8"))
    sections = {k: raw.get(k) for k in _REPORT_SECTIONS} if isinstance(raw, dict) else {}
    drawings = {}
    for drawing in project.atlas.drawings:
        path = folder / drawing.image
        drawings[drawing.image] = _sha256_input(path) if path.is_file() else None
    canonical = yaml.safe_dump({"sections": sections, "drawings": drawings}, sort_keys=True,
                               allow_unicode=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _drop_field(text: str, at: int) -> str:
    """``text`` (CSV) with field ``at`` of every record cut out together with
    one comma beside it; every other character (BOM, quotes, spaces, line
    ends, blank lines) kept as it is. A record with no field ``at`` is kept."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        start, fstart, quoted = i, i, False
        fields: list[tuple[int, int]] = []
        while i < n:
            ch = text[i]
            if quoted:
                if ch == '"' and text[i + 1:i + 2] == '"':
                    i += 2
                    continue
                quoted = ch != '"'
            elif ch == '"' and i == fstart:
                quoted = True
            elif ch == ",":
                fields.append((fstart, i))
                fstart = i + 1
            elif ch in "\r\n":
                break
            i += 1
        fields.append((fstart, i))
        end = i
        if text[i:i + 2] == "\r\n":
            i += 2
        elif i < n:
            i += 1
        if len(fields) > at:
            cut = (fields[at - 1][1], fields[at][1]) if at else (fields[0][0], fields[1][0] if len(fields) > 1 else end)
            out.append(text[start:cut[0]] + text[cut[1]:i])
        else:
            out.append(text[start:i])
    return "".join(out)


def materials_digest(path: str | pathlib.Path) -> str:
    """sha256 of materials.csv as the calculations read it (calculator spec
    D3): its raw bytes, with the `vlt` field (and its comma) cut out of
    every record when the header has that column. BOM, quoting, spaces and
    line ends are kept, so adding or changing a VLT, by hand or through
    ``inputs``, never marks a run out of date: `vlt` changes no calculation.
    Every other edit reads as changed, as before."""
    data = read_input_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return hashlib.sha256(data).hexdigest()
    header = next(csv.reader(io.StringIO(text.removeprefix("\ufeff"), newline="")), [])
    names = [cell.strip() for cell in header]
    if "vlt" not in names:
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(_drop_field(text, names.index("vlt")).encode("utf-8")).hexdigest()


def _key(path: pathlib.Path, folder: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(folder.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def input_fingerprints(project: Project, folder: pathlib.Path) -> dict[str, str]:
    """sha256 of project.yaml (without its ``report``, ``atlas`` and ``calculator`` sections), materials.csv,
    spaces.csv, the credit rules, the weather file, every scene file and,
    when it exists, splits.csv, and when project.yaml names one, the
    sensor points file, keyed by path, sorted."""
    folder = pathlib.Path(folder)
    paths = [
        folder / "materials.csv", folder / "spaces.csv",
        pathlib.Path(project.credit_rules_path), pathlib.Path(project.weather_file),
        *(pathlib.Path(f) for f in project.scene.files),
    ]
    if (folder / SPLITS_FILE).exists():
        paths.append(folder / SPLITS_FILE)
    if project.grid.points_file is not None:
        paths.append(pathlib.Path(project.grid.points_file))
    prints = {_key(p, folder): _sha256_input(p) for p in paths}
    prints[_key(folder / "materials.csv", folder)] = materials_digest(folder / "materials.csv")
    prints[_key(folder / "project.yaml", folder)] = _project_yaml_digest(folder / "project.yaml")
    return dict(sorted(prints.items()))


def stale_inputs(recorded: Mapping[str, str], project: Project, folder: pathlib.Path) -> list[str]:
    """Every key whose digest differs between ``recorded`` and the disk,
    including a key present on only one side; sorted."""
    current = input_fingerprints(project, folder)
    return sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
