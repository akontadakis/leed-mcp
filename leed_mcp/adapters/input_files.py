# leed_mcp/adapters/input_files.py
"""Reading and writing the input files of the ``inputs`` tool (spec 6.3).

Every read refuses a link from the project folder down to the file
(``checked_below`` and the no-follow ``open_plain``) and reads the bytes once; the digest, the content and
any check all come from those bytes. Every write follows spec 6.3's rules:

1. ``expected_sha256`` must equal the file on disk (None: the file must
   not exist), else ``INPUT_CHANGED_ON_DISK`` with this server's reason
   (spec 13, N1).
2. The write holds the pipeline's folder lock (``folder_lock(folder,
   "inputs")``), so it never lands while a job reads the inputs:
   ``LOCK_HELD``. The job index lock of spec 8.1 joins it in plan 4.
3. The new file is written as ``<name>.part`` and checked before it
   replaces the old one (``edit.files.save_files``, with a new ``Served``
   per call, spec 4.1).
4. The old file is backed up to ``_run/backup/`` (newest 20 kept).

Every write also refuses a path that project.yaml names outside the
project folder (spec 10), because the file can be edited by hand.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
import pathlib
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import yaml

from leed_daylight.constants import FILL_ME
from leed_daylight.edit.files import BACKUP_DIR, ChangedOutside, Served, save_files
from leed_daylight.engine.jobs import append_private, checked_below, checked_dir, folder_lock, open_plain
from leed_daylight.scene import input_file
from leed_daylight.scene.input_file import InputFileError, too_large
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused, one_line

__all__ = [
    "INPUT_LOG", "YAML_NODES_MAX", "YAMLTooLarge", "Written", "as_file", "check_cell_text", "check_project_paths",
    "digest", "jsonable", "load_yaml", "log_source",
    "outside_paths", "read_bytes", "read_project_raw", "write_file",
]

#: The log of every project.yaml update's ``source`` (spec 6.3).
INPUT_LOG = ("_run", "input_log.jsonl")


#: A text cell starting with one of these is read as a formula by a
#: spreadsheet (CSV formula injection), so it is refused.
_FORMULA_START = ("=", "+", "-", "@")
#: Unicode categories a CSV text cell refuses: control characters (``\\n``,
#: ``\\r``, ``\\t``, NUL, ``\\x85`` ...) and the line and paragraph separators.
_BAD_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


def check_cell_text(value: str, where: str, column: str) -> str:
    """Refuse (``INPUT_INVALID``, field ``rows``) a text cell for a CSV
    input file that holds a line break or control character, ``<!--``
    (an auto-block marker downstream), or starts (after blanks) with
    ``=``, ``+``, ``-`` or ``@``. Returns ``value``."""
    bad = [ch for ch in value if unicodedata.category(ch) in _BAD_CATEGORIES]
    if bad:
        raise Refused(refusal("INPUT_INVALID", f"{where}: {column} holds a line break or control character "
                              f"(U+{ord(bad[0]):04X}); write it on one line without one.", field="rows"))
    if "<!--" in value:
        raise Refused(refusal("INPUT_INVALID", f"{where}: {column} holds '<!--'; write it without.", field="rows"))
    lead = value.strip()[:1]
    if lead and lead in _FORMULA_START:
        raise Refused(refusal("INPUT_INVALID", f"{where}: {column} starts with {lead!r}, which a spreadsheet reads "
                              f"as a formula; write it without the leading {lead!r}.", field="rows"))
    return value


def read_bytes(folder: pathlib.Path, name: str) -> bytes | None:
    """The bytes of ``<folder>/<name>``, or None when there is no such file.
    A link anywhere from ``folder`` down is refused (``UnsafePath``): the
    folders by ``checked_below``, the file by the no-follow open of
    ``open_plain`` (as ``results_files._read_guarded``), so a link put in
    place after a check is not followed either. A FIFO and a folder are
    refused (``UnsafePath``, never waited on), and a file over the input
    size cap ``INPUT_FILE_MAX`` (``InputFileError``), before and after the
    read, as ``read_input_bytes`` refuses them."""
    path = folder / name
    checked_below(folder, path.parent)
    limit = input_file.INPUT_FILE_MAX
    try:
        handle = open_plain(path)
    except FileNotFoundError:
        return None
    with handle:
        if os.fstat(handle.fileno()).st_size > limit:
            raise too_large(path, limit)
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise too_large(path, limit)
    return data


def digest(data: bytes | None) -> str | None:
    """The sha256 of ``data``, or None for a file that does not exist."""
    return None if data is None else hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def as_file(folder: pathlib.Path, name: str, data: bytes) -> Iterator[pathlib.Path]:
    """``data`` as a file named ``name`` in a new temporary folder outside
    the project, for a pipeline reader that takes a path, so a check sees
    the bytes already read. A message that names the temporary folder is
    rewritten to name ``folder``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / name
        path.write_bytes(data)
        real = os.path.realpath(tmp)

        def rewrite(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            # The real path first: on macOS /private/var/... holds /var/....
            return value.replace(real, str(folder)).replace(tmp, str(folder))

        try:
            yield path
        except (ValueError, OSError) as exc:
            exc.args = tuple(rewrite(arg) for arg in exc.args)
            if isinstance(exc, OSError):
                exc.filename, exc.filename2 = rewrite(exc.filename), rewrite(exc.filename2)
            raise


#: The most nodes a YAML input file may expand to, an alias counted as
#: the whole node it stands for each time it is used. Chosen: far above
#: any real project.yaml or credit_rules.yaml, far below a "billion laughs".
YAML_NODES_MAX = 100_000


class YAMLTooLarge(yaml.YAMLError):
    """A YAML document that expands to more than ``YAML_NODES_MAX`` nodes,
    holds a recursive alias, or is nested too deeply to read."""


def _children(node: yaml.Node) -> list[yaml.Node]:
    if isinstance(node, yaml.SequenceNode):
        return list(node.value)
    if isinstance(node, yaml.MappingNode):
        return [part for pair in node.value for part in pair]
    return []


def _check_expanded_size(root: yaml.Node) -> None:
    """Refuse (``YAMLTooLarge``) a composed document whose expanded size is
    over ``YAML_NODES_MAX`` or that holds a cycle. Each node's expanded
    size is computed once (memoised by identity) on the composed graph, so
    nothing is expanded; the walk is iterative, so depth costs no stack."""
    sizes: dict[int, int] = {}
    on_path: set[int] = set()
    stack: list[tuple[yaml.Node, bool]] = [(root, False)]
    while stack:
        node, done = stack.pop()
        key = id(node)
        if done:
            on_path.discard(key)
            sizes[key] = 1 + sum(sizes[id(child)] for child in _children(node))
            if sizes[key] > YAML_NODES_MAX:
                raise YAMLTooLarge(f"the YAML expands to more than {YAML_NODES_MAX} nodes (an alias counts as "
                                   "the whole node it stands for each time); it is not read")
            continue
        if key in sizes:
            continue
        if key in on_path:
            raise YAMLTooLarge("the YAML holds a recursive alias (an anchor used inside itself); it is not read")
        on_path.add(key)
        stack.append((node, True))
        stack.extend((child, False) for child in _children(node) if id(child) not in sizes)


def load_yaml(text: str) -> Any:
    """``yaml.safe_load(text)``, refusing (``YAMLTooLarge``, a
    ``yaml.YAMLError``) a document that expands to more than
    ``YAML_NODES_MAX`` nodes, holds a recursive alias or is nested too
    deeply. None for an empty document."""
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_expanded_size(node)
        return loader.construct_document(node)
    except RecursionError:
        raise YAMLTooLarge("the YAML is nested too deeply; it is not read") from None
    finally:
        loader.dispose()


def jsonable(value: Any) -> Any:
    """A loaded YAML value as JSON: keys as text, dates as ISO text, a
    float that is not finite as None, bytes (``!!binary``) as None, a set
    as a sorted list (sorted as text when the items do not compare)."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value)
        except TypeError:
            items = sorted(value, key=str)
        return [jsonable(v) for v in items]
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_project_raw(folder: pathlib.Path) -> tuple[bytes | None, dict[str, Any] | None]:
    """``(bytes, mapping)`` of project.yaml; ``(None, None)`` without one.
    Raises ``ValueError`` for a file that is not a YAML mapping and a
    ``yaml.YAMLError`` (``YAMLTooLarge`` among them) for one that is not
    YAML or is refused by ``load_yaml``."""
    data = read_bytes(folder, "project.yaml")
    if data is None:
        return None, None
    raw = load_yaml(data.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("project.yaml: project.yaml must be a mapping at the top level")
    return data, raw


def _named_paths(raw: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(key, value)`` of every path project.yaml names (spec 10) that is
    text and not FILL_ME."""
    def get(*keys):
        value: Any = raw
        for key in keys:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    named = [("weather.file", get("weather", "file")), ("standard.credit_rules", get("standard", "credit_rules")),
             ("grid.points_file", get("grid", "points_file"))]
    files = get("scene", "files")
    if isinstance(files, list):
        named += [(f"scene.files[{i}]", f) for i, f in enumerate(files)]
    drawings = get("atlas", "drawings")
    if isinstance(drawings, list):
        named += [(f"atlas.drawings[{i}].image", d.get("image")) for i, d in enumerate(drawings)
                  if isinstance(d, Mapping)]
    return [(key, value) for key, value in named
            if isinstance(value, str) and value.strip() and value.strip() != FILL_ME]


def outside_paths(folder: pathlib.Path, raw: Mapping[str, Any]) -> list[str]:
    """The keys of project.yaml ``raw`` whose path resolves outside
    ``folder`` (symlinks followed), as ``key: path``."""
    out = []
    for key, value in _named_paths(raw):
        resolved = (folder / value).resolve()
        if resolved != folder and folder not in resolved.parents:
            out.append(f"{key}: {resolved}")
    return out


def check_project_paths(folder: pathlib.Path, raw: Mapping[str, Any] | None) -> None:
    """Refuse (``PATH_OUTSIDE_ROOTS``) when project.yaml ``raw`` names a
    path outside the project folder (spec 10)."""
    outside = outside_paths(folder, raw or {})
    if outside:
        raise Refused(refusal(
            "PATH_OUTSIDE_ROOTS",
            f"project.yaml names {len(outside)} path(s) outside the project folder ({folder}): "
            f"{'; '.join(outside)}. Every file it names must be inside the folder; change "
            "the path first (inputs action='update', file='project')."))


class Written(NamedTuple):
    """A file just saved: its path, the sha256 of its new bytes, and the
    backup of the old file (None when there was none)."""

    path: pathlib.Path
    sha256: str
    backup: pathlib.Path | None


def _changed(name: str) -> Refused:
    return Refused(refusal(
        "INPUT_CHANGED_ON_DISK",
        f"{name} changed on disk since it was read; read it again with inputs(action='read') and "
        "send its new sha256 as expected_sha256.", field="expected_sha256"))


def check_expected(name: str, data: bytes | None, expected: str | None) -> None:
    """Refuse (``INPUT_CHANGED_ON_DISK``) unless ``expected`` is the digest
    of ``data``: None means the file must not exist yet."""
    if digest(data) != expected:
        if data is None:
            raise Refused(refusal(
                "INPUT_CHANGED_ON_DISK",
                f"{name} does not exist; send expected_sha256 null (or leave it out) to create it.",
                field="expected_sha256"))
        if expected is None:
            raise Refused(refusal(
                "INPUT_CHANGED_ON_DISK",
                f"{name} already exists; read it with inputs(action='read') and send its sha256 as "
                "expected_sha256 to replace it.", field="expected_sha256"))
        raise _changed(name)


def _backups(folder: pathlib.Path, name: str) -> set[pathlib.Path]:
    """The backups of ``name`` now; a sub-folder of ``name`` is kept under
    ``_run/backup/`` (``edit.files.backup``)."""
    rel = pathlib.PurePosixPath(name)
    where = checked_dir(folder, *BACKUP_DIR, *rel.parent.parts, make=False)
    if not where.is_dir():
        return set()
    return {p for p in where.iterdir() if p.name.startswith(f"{rel.name}.")}


def write_file(folder: pathlib.Path, name: str, expected: str | None,
               build: Callable[[bytes | None, Mapping[str, Any] | None], bytes],
               check: Callable[[pathlib.Path], None]) -> Written:
    """Save ``<folder>/<name>`` under spec 6.3's rules (module docstring).

    Inside the folder lock: project.yaml's paths are checked; the old bytes
    are read once and compared with ``expected``; ``build(old, raw)`` makes
    the new bytes from them (``raw`` is project.yaml, loaded); ``check``
    runs on the ``.part`` file; the old file is backed up and replaced.
    ``save_files`` reads the file once more for its own change check."""
    with folder_lock(folder, "inputs"):
        if name == "project.yaml":
            old, raw = read_project_raw(folder)
        else:
            try:
                _, raw = read_project_raw(folder)
            except InputFileError:
                raise  # not a plain file, or too large: PATH_UNSAFE (errors.MAPPING)
            except (ValueError, yaml.YAMLError) as exc:
                raise Refused(refusal(
                    "INPUT_INVALID", f"project.yaml cannot be read, so the paths it names cannot be "
                    f"checked (spec 10); fix it first: {one_line(exc)}")) from None
            check_project_paths(folder, raw)
            old = read_bytes(folder, name)
        check_expected(name, old, expected)
        data = build(old, raw)
        served = Served()
        if served.remember(folder, name) != expected:
            raise _changed(name)
        before = _backups(folder, name)
        try:
            save_files(folder, {name: data}, check=lambda parts: check(parts[name]), served=served)
        except ChangedOutside:
            raise _changed(name) from None
        new = sorted(_backups(folder, name) - before)
    return Written(folder / name, digest(data), new[-1] if new else None)


def log_source(folder: pathlib.Path, *, name: str, keys: Sequence[str], unset: Sequence[str],
               source: str) -> pathlib.Path:
    """Append one line to ``_run/input_log.jsonl`` (spec 6.3): the time
    (UTC), the file, the keys set and removed, and the ``source`` sentence."""
    path = checked_dir(folder, *INPUT_LOG[:-1]) / INPUT_LOG[-1]
    line = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "file": name, "set": list(keys), "unset": list(unset), "source": source}
    append_private(path, json.dumps(line, ensure_ascii=False) + "\n")
    return path
