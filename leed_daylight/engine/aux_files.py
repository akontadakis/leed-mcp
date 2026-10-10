"""The auxiliary files a scene reaches through ``RAYPATH`` (``.cal``,
``.dat``, picture and font files), so a matrix fingerprint can notice when
one of them changes: the prepared ``.rad`` files only *name* them.
"""
from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterable, Sequence

from ..scene.primitives import Primitive

#: Primitive types whose string arguments can name a file: the function
#: (``...func``), data (``...data``), picture (``...pict``) and text
#: (``...text``, a font) kinds.
_FILE_SUFFIXES = ("func", "data", "pict", "text")


def raypath_dirs(project, model_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The extra ``RAYPATH`` directories for the real runner: every scene
    file's own folder (in ``scene.files`` order, each once), then the
    prepared model folder. Radiance commands run in this process's working
    directory, so a scene's relative ``.cal``/``.xml``/data reference
    resolves only through these."""
    dirs: list[pathlib.Path] = []
    for path in (*(pathlib.Path(f).parent for f in project.scene.files), model_dir):
        resolved = pathlib.Path(path).resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    return tuple(dirs)


def _is_number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _named_files(primitives: Iterable[Primitive]) -> list[str]:
    """Every string argument of a function/data/picture/text primitive that
    looks like a file name: not ``.`` (no file), not a number, and with an
    extension (function and variable names have none)."""
    names: dict[str, None] = {}
    for prim in primitives:
        if not prim.ptype.endswith(_FILE_SUFFIXES):
            continue
        for word in prim.sargs:
            if word != "." and not _is_number(word) and "." in pathlib.PurePath(word).name:
                names[word] = None
    return sorted(names)


def aux_file_hashes(
    primitives: Iterable[Primitive], search_dirs: Sequence[pathlib.Path],
) -> dict[str, str]:
    """``{name: sha256}`` for each file the primitives name, found the way
    Radiance finds it: as given when absolute, else in each of
    ``search_dirs`` in turn. A name that is not found maps to ``"missing"``
    -- the name alone still goes into the fingerprint, and Radiance fails on
    it later."""
    out: dict[str, str] = {}
    for name in _named_files(primitives):
        candidates = [pathlib.Path(name)]
        if not candidates[0].is_absolute():
            candidates = [pathlib.Path(d) / name for d in search_dirs]
        out[name] = "missing"
        for candidate in candidates:
            try:
                out[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            break
    return out
