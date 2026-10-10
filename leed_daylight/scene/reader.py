"""Parse a Radiance scene into primitives, expanding !xform.

An ``alias`` is resolved here into a copy of the modifier it names, so
nothing downstream ever sees one.

Only the !xform options a daylight model actually uses are supported.  Anything
else is refused by name, because silently ignoring a transform would move
geometry without any error appearing.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
import shlex
from collections.abc import Sequence

import numpy as np

from ..units import to_metres
from .input_file import read_input_text
from .primitives import SURFACE_TYPES, Primitive

SUPPORTED = {"-t", "-rx", "-ry", "-rz", "-s", "-mx", "-my", "-mz", "-n"}

#: Types xform's ``-n`` prefix renames.  Radiance's xform.c prefixes only
#: the names of objects that are not modifiers (surfaces, sources,
#: instances); a material, pattern or texture keeps its name, and the
#: modifier field of every line is never touched.
_PREFIXED_TYPES = SURFACE_TYPES | {"source", "instance"}


#: How many !xform includes may nest, and how many primitives one read may
#: produce in all. Chosen far above any real building scene; they stop a
#: chain of includes that doubles at every level.
MAX_INCLUDE_DEPTH = 32
MAX_PRIMITIVES = 2_000_000
#: How many files one read may open in all, counting every !xform expansion
#: (a file included twice counts twice), empty files too.
MAX_INCLUDE_READS = 10_000


class XformError(ValueError):
    """An inline command this reader will not expand."""


def _identity() -> np.ndarray:
    return np.eye(4)


def _translate(dx, dy, dz) -> np.ndarray:
    m = _identity()
    m[:3, 3] = (dx, dy, dz)
    return m


def _rotate(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = _identity()
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    m[a, a] = c; m[a, b] = -s; m[b, a] = s; m[b, b] = c
    return m


def _scale(factor: float) -> np.ndarray:
    m = _identity()
    m[0, 0] = m[1, 1] = m[2, 2] = factor
    return m


def _mirror(axis: int) -> np.ndarray:
    m = _identity()
    m[axis, axis] = -1.0
    return m


def _parse_xform(tokens: list[str]) -> tuple[np.ndarray, str, list[str]]:
    """Return (4x4 matrix, name prefix, file arguments).

    Radiance's xform takes any number of files after its options and
    transforms each with the same matrix and prefix, in order; so does
    this reader.
    """
    matrix, prefix, i = _identity(), "", 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("-"):
            break
        if token not in SUPPORTED:
            raise XformError(
                f"!xform option {token} is not supported; supported options are "
                + " ".join(sorted(SUPPORTED))
            )
        if token == "-t":
            matrix = _translate(*(float(v) for v in tokens[i + 1:i + 4])) @ matrix
            i += 4
        elif token in ("-rx", "-ry", "-rz"):
            matrix = _rotate("xyz".index(token[2]), float(tokens[i + 1])) @ matrix
            i += 2
        elif token == "-s":
            matrix = _scale(float(tokens[i + 1])) @ matrix
            i += 2
        elif token in ("-mx", "-my", "-mz"):
            matrix = _mirror("xyz".index(token[2])) @ matrix
            i += 1
        else:                                            # -n
            prefix = tokens[i + 1]
            i += 2
    if i >= len(tokens):
        raise XformError("!xform has no file argument")
    return matrix, prefix, tokens[i:]


def _apply(matrix: np.ndarray, fargs: tuple[float, ...]) -> tuple[float, ...]:
    pts = np.asarray(fargs, dtype=np.float64).reshape(-1, 3)
    out = (matrix[:3, :3] @ pts.T).T + matrix[:3, 3]
    if np.linalg.det(matrix[:3, :3]) < 0:
        out = out[::-1]                                  # keep the normal direction
    return tuple(out.reshape(-1))


_QUOTES = "\"'"


def _words(code: str, path: pathlib.Path, lineno: int) -> list[str]:
    """Split one line of primitive text into Radiance words.

    Words are whitespace separated; a word that opens with a double or a
    single quote runs to the matching quote and may hold spaces (the
    quotes are dropped), as Radiance's own word reader allows.  A quote
    anywhere else in a word is an ordinary character (``it's``), as in
    Radiance's ``fgetword``.  A ``#`` outside quotes ends the line.  An
    unterminated quote is refused with the file and line, never guessed at.
    """
    words: list[str] = []
    k, n = 0, len(code)
    while k < n:
        c = code[k]
        if c.isspace():
            k += 1
        elif c == "#":
            break
        elif c in _QUOTES:
            close = code.find(c, k + 1)
            if close < 0:
                raise ValueError(
                    f"{path}, line {lineno}: unterminated {c} quote in a primitive"
                )
            words.append(code[k + 1:close])
            k = close + 1
        else:
            start = k
            while k < n and not code[k].isspace() and code[k] != "#":
                k += 1
            words.append(code[start:k])
    return words


def quote_word(word: str) -> str:
    """``word`` as it must be written for :func:`read_scene` (and Radiance)
    to read it back as the same single word.

    A word holding whitespace or a ``#``, one starting with a quote, one
    holding a quote, or an empty word is wrapped in whichever quote
    character it does not contain; any other word is written as it is.
    A word that must be quoted but holds both quote characters cannot be
    written at all and is refused by name.
    """
    special = (
        not word
        or any(c.isspace() for c in word)
        or "#" in word
        or word[0] in _QUOTES
    )
    if not special and not any(q in word for q in _QUOTES):
        return word
    for quote in _QUOTES:
        if quote not in word:
            return f"{quote}{word}{quote}"
    if not special:
        return word  # quotes only inside the word: ordinary characters
    raise ValueError(
        f"primitive word {word!r} holds whitespace, a '#' or a leading "
        "quote and both quote characters; no Radiance quoting can hold it"
    )


def _tokens_of(text: str, path: pathlib.Path):
    """Yield ``(kind, token, line)``: ``"!"`` for a command, ``"t"`` for a word.

    A command line ending in a backslash continues on the next line, as
    in a Radiance scene file.
    """
    lines = text.splitlines()
    k = 0
    while k < len(lines):
        lineno, line = k + 1, lines[k]
        k += 1
        stripped = line.strip()
        if stripped.startswith("!"):
            command = stripped[1:]
            while command.endswith("\\") and k < len(lines):
                command = command[:-1] + " " + lines[k].strip()
                k += 1
            yield ("!", command, lineno)
            continue
        for token in _words(line, path, lineno):
            yield ("t", token, lineno)


def _alias(
    path: pathlib.Path, modifier: str, name: str, target: str, out: list[Primitive]
) -> Primitive:
    """Resolve ``modifier alias name target`` to a copy of ``target``.

    Radiance's alias has no argument counts: it names one earlier modifier
    and makes ``name`` an exact copy of it, with its own modifier replaced
    by ``modifier`` unless that is ``void``.  The copy is what is returned,
    so every stage below this reader sees an ordinary material and never
    an alias.  An alias is a modifier, so ``-n`` never prefixes it.
    """
    for prim in reversed(out):
        if prim.name == target and not prim.is_surface:
            return dataclasses.replace(
                prim,
                name=name,
                modifier=prim.modifier if modifier == "void" else modifier,
                source=str(path),
            )
    raise ValueError(
        f"{path}: alias {name!r} refers to {target!r}, which is not defined "
        "as a modifier before it"
    )


def _read_file(path: pathlib.Path, matrix, prefix, out: list[Primitive], depth: int = 0,
               reads: list[int] | None = None) -> None:
    if reads is None:
        reads = [0]
    reads[0] += 1
    if reads[0] > MAX_INCLUDE_READS:
        raise XformError(
            f"{path}: the scene reads more than {MAX_INCLUDE_READS:,} files once the !xform includes "
            "are expanded; remove repeated includes or split the model"
        )
    if depth > MAX_INCLUDE_DEPTH:
        raise XformError(
            f"{path}: the !xform includes nest too deep (over {MAX_INCLUDE_DEPTH} levels); "
            "a scene file probably includes itself; remove that !xform"
        )
    if not path.exists():
        raise FileNotFoundError(f"scene file not found: {path}")
    stream = list(_tokens_of(read_input_text(path), path))
    i = 0

    def take(n: int) -> list[str]:
        nonlocal i
        values = []
        while len(values) < n:
            if i >= len(stream):
                raise ValueError(f"{path}: ran out of tokens inside a primitive")
            kind, token, lineno = stream[i]
            i += 1
            if kind == "!":
                raise ValueError(
                    f"{path}, line {lineno}: a command appeared inside a primitive"
                )
            values.append(token)
        return values

    def take_count() -> int:
        token = take(1)[0]
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(
                f"{path}: {token!r} is not a valid argument count inside a primitive"
            ) from exc

    while i < len(stream):
        if len(out) >= MAX_PRIMITIVES:
            raise XformError(
                f"{path}: the scene holds more than {MAX_PRIMITIVES:,} primitives once the "
                "!xform includes are expanded; remove repeated includes or split the model"
            )
        kind, token, lineno = stream[i]
        if kind == "!":
            i += 1
            try:
                parts = shlex.split(token)
            except ValueError as exc:
                raise XformError(
                    f"{path}, line {lineno}: cannot read the command "
                    f"{('!' + token)!r}: {exc}"
                ) from None
            if not parts or parts[0] != "xform":
                raise XformError(f"{path}: only !xform is expanded, found: {token}")
            try:
                sub_matrix, sub_prefix, targets = _parse_xform(parts[1:])
            except XformError as exc:
                raise XformError(f"{path}: {exc}") from exc
            except (ValueError, TypeError, IndexError) as exc:
                raise XformError(
                    f"{path}: cannot read the !xform options in {('!' + token)!r}"
                ) from exc
            joined = f"{prefix}.{sub_prefix}" if prefix and sub_prefix else (prefix or sub_prefix)
            for target in targets:
                _read_file(path.parent / target, matrix @ sub_matrix, joined, out, depth + 1, reads)
            continue
        modifier, ptype, name = take(3)
        if ptype == "alias":
            out.append(_alias(path, modifier, name, take(1)[0], out))
            continue
        sargs = tuple(take(take_count()))
        iargs = tuple(int(v) for v in take(take_count()))
        fargs = tuple(float(v) for v in take(take_count()))
        full = f"{prefix}.{name}" if prefix and ptype in _PREFIXED_TYPES else name
        prim = Primitive(modifier, ptype, full, sargs, iargs, fargs, str(path))
        if ptype == "polygon":
            prim = dataclasses.replace(prim, fargs=_apply(matrix, fargs))
        elif prim.is_surface and not np.allclose(matrix, np.eye(4), atol=1e-12):
            raise XformError(
                f"{full}: a {ptype} inside an !xform is not supported; this reader "
                "transforms polygon geometry only. Move it outside the transform, "
                "or convert it to polygons."
            )
        out.append(prim)


def read_scene(paths: Sequence[str | pathlib.Path], units: str) -> list[Primitive]:
    """Read one or more .rad files into a flat list of primitives.

    `units` is required, with no default: a missing unit is a refusal, not a
    guess.  Every polygon's real arguments are converted to metres exactly
    once here, after the whole !xform matrix has already been applied, so
    below this function every polygon length in the package is metres.  A
    non-polygon surface's real arguments are left exactly as read: this
    reader converts polygon geometry only, and an unsupported surface type
    is refused by name at validation rather than scaled.
    """
    factor = to_metres(1.0, units)
    out: list[Primitive] = []
    reads = [0]
    for path in paths:
        try:
            _read_file(pathlib.Path(path), _identity(), "", out, 0, reads)
        except RecursionError:
            # A scene file that includes itself, directly or through others
            # (security review L2): one plain refusal, not a crash.
            raise XformError(
                f"{path}: the !xform includes nest too deep; a scene file probably includes itself; "
                "remove that !xform"
            ) from None
    return [
        dataclasses.replace(p, fargs=tuple(v * factor for v in p.fargs))
        if p.ptype == "polygon" else p
        for p in out
    ]
