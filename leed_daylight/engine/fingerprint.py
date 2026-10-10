# leed_daylight/engine/fingerprint.py
"""Input fingerprints so a stopped run resumes and a stale run recomputes
(layer 4; spec 4.1, spec 6.13).

A ``Fingerprint`` is what one engine step (a sky matrix, a daylight
coefficient matrix, ...) was built from: the SHA-256 of every input file it
read, the settings that change its result, and the tool version that
produced it. ``fingerprint_of`` builds one; ``write_sidecar`` records it
beside the output it describes, as ``<output>.fingerprint.json``;
``is_current`` says whether that output can be trusted as-is or must be
rebuilt.

Hashing reads bytes, in chunks (a sky matrix can be hundreds of MB), never
the modification time: copying a project folder to a new location changes
every mtime and must not force a rebuild of everything in it. Only a real
content change, a changed parameter, or a changed tool version makes a
fingerprint stale -- rebuilding after a tool upgrade is cheap; a wrong
number that slips through because the tool changed and the cache did not
know is not.

``is_current`` never raises: a missing output, a missing or unreadable or
malformed sidecar, a different set of input labels, a different hash, a
different parameter, or a different tool version are all just "stale",
the same as an ordinary changed file. Only building the fingerprint itself
can refuse -- a missing input file, a directory given as an input, or a
parameter that cannot be written to JSON are each a named error, because a
silent "stale" there would hide a caller's mistake instead of a real
recompute.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass

from ..constants import TOOL_VERSION

#: Read files this many bytes at a time when hashing, so a multi-hundred-MB
#: sky matrix never has to fit in memory twice over.
_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB

#: The sidecar's suffix, appended to the output's own name (not replacing
#: it): ``result.dat`` -> ``result.dat.fingerprint.json``.
_SIDECAR_SUFFIX = ".fingerprint.json"

#: JSON-representable leaf types a parameter value may hold, besides the
#: container types (list/tuple/dict) handled recursively below.
_JSON_LEAF_TYPES = (str, int, bool, type(None))


class FingerprintError(ValueError):
    """A fingerprint's inputs or parameters cannot be computed as given."""


@dataclass(frozen=True)
class Fingerprint:
    """What one engine step's output was built from.

    ``inputs`` maps a caller-chosen label (e.g. ``"scene"``, ``"manifest"``)
    to that file's SHA-256 hex digest. ``parameters`` holds the settings
    that change the result -- already normalised (tuples become lists) so
    that a fingerprint compares equal to itself after a round trip through
    its sidecar. ``tool_version`` is always ``constants.TOOL_VERSION`` at
    the time ``fingerprint_of`` ran; it is a field (not read fresh by
    ``is_current``) so a test can build a stale one with
    ``dataclasses.replace``.
    """

    inputs: dict[str, str]
    parameters: dict[str, object]
    tool_version: str


def _hash_file(label: str, path: pathlib.Path) -> str:
    """The SHA-256 hex digest of ``path``'s bytes, read in chunks.

    A directory is a named :class:`FingerprintError` (naming the label and
    the path): a fingerprint is over file bytes, and a directory has none
    of its own to hash. A missing path is a ``FileNotFoundError`` naming
    the label and the path -- not this module's own error type, because
    the task's interface calls for the builtin here specifically, so a
    caller can catch a plain "file went missing" the same way it would
    anywhere else in the tool.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        raise FingerprintError(
            f"input {label!r} is a directory, not a file: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(f"input {label!r} not found: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalise_parameter(value: object, *, where: str) -> object:
    """Validate and normalise one parameter value (recursively).

    Accepts str, int, bool, None, a finite float, and lists/tuples/dicts
    (with str keys) of those, following the same rule recursively. Tuples
    become lists. Anything else -- a ``pathlib.Path``, a numpy array, NaN
    or infinity, a non-string dict key -- is a named :class:`FingerprintError`
    naming where in the parameter tree it was found and what it was.
    """
    if isinstance(value, _JSON_LEAF_TYPES):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError(
                f"parameter {where} is not finite: {value!r}"
            )
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalise_parameter(item, where=f"{where}[{i}]")
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalised: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FingerprintError(
                    f"parameter {where} has a non-string key: {key!r}"
                )
            normalised[key] = _normalise_parameter(item, where=f"{where}[{key!r}]")
        return normalised
    raise FingerprintError(
        f"parameter {where} is not JSON-representable: "
        f"{type(value).__name__} {value!r}"
    )


def fingerprint_of(
    paths: Mapping[str, pathlib.Path], parameters: Mapping[str, object]
) -> Fingerprint:
    """Build a :class:`Fingerprint` over ``paths`` (label -> file) and
    ``parameters`` (the settings that change the result).

    Every path is hashed immediately (a missing or directory input refuses
    right away, not later when something tries to use a half-built
    fingerprint). ``tool_version`` is always the running tool's own
    ``constants.TOOL_VERSION``; callers never pass one in.
    """
    inputs: dict[str, str] = {}
    for label, path in paths.items():
        inputs[label] = _hash_file(label, path)

    normalised_parameters = _normalise_parameter(dict(parameters), where="parameters")

    return Fingerprint(
        inputs=inputs,
        parameters=normalised_parameters,
        tool_version=TOOL_VERSION,
    )


def _sidecar_path(output: pathlib.Path) -> pathlib.Path:
    output = pathlib.Path(output)
    return output.parent / f"{output.name}{_SIDECAR_SUFFIX}"


def write_sidecar(output: pathlib.Path, fp: Fingerprint) -> None:
    """Write ``fp`` beside ``output`` as ``<output>.fingerprint.json``.

    Written atomically: a temporary file in the same folder, then
    ``os.replace``, so a crash mid-write never leaves a half-written (and
    therefore falsely "current" or falsely corrupt) sidecar behind. Keys
    are sorted and ``allow_nan`` is off, so the file is both diffable and
    never silently carries a NaN that ``is_current`` would have to guess
    about.
    """
    sidecar = _sidecar_path(output)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "inputs": fp.inputs,
        "parameters": fp.parameters,
        "tool_version": fp.tool_version,
    }

    fd, tmp_name = tempfile.mkstemp(
        dir=sidecar.parent, prefix=f".{sidecar.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
        os.replace(tmp_name, sidecar)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def delete_sidecar(output: pathlib.Path) -> None:
    """Remove ``output``'s fingerprint sidecar, if any (final-fix M1).

    Call this BEFORE rebuilding ``output`` in place (truncating and
    rewriting it), never after: a crash between the truncation and
    ``write_sidecar`` must never leave a freshly-truncated (or partially
    written) output sitting beside an OLD sidecar that a later run, still
    using the old parameters, would wrongly accept as current. Missing is
    not an error -- the first build of an output has no sidecar yet.
    """
    sidecar = _sidecar_path(output)
    sidecar.unlink(missing_ok=True)


def is_current(output: pathlib.Path, fp: Fingerprint) -> bool:
    """Whether ``output`` can be trusted as-is against fingerprint ``fp``.

    False when: ``output`` itself is missing (even if a sidecar exists);
    the sidecar is missing, unreadable, or not valid JSON (stale, not a
    crash); the set of input labels differs; any input hash differs; any
    parameter differs; or the tool version differs. True only when every
    one of those checks passes.
    """
    output = pathlib.Path(output)
    if not output.exists():
        return False

    sidecar = _sidecar_path(output)
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        stored = json.loads(raw)
    except json.JSONDecodeError:
        return False

    if not isinstance(stored, dict):
        return False

    stored_inputs = stored.get("inputs")
    if not isinstance(stored_inputs, dict):
        return False
    if set(stored_inputs.keys()) != set(fp.inputs.keys()):
        return False
    if any(stored_inputs[label] != digest for label, digest in fp.inputs.items()):
        return False

    if stored.get("parameters") != fp.parameters:
        return False

    if stored.get("tool_version") != fp.tool_version:
        return False

    return True
