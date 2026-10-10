# leed_daylight/project/materials.py
"""Load materials.csv: every modifier the scene uses gets exactly one role.

The scene reader hands back Radiance primitives with a modifier name and
nothing else; it has no notion of "this is glazing" or "this is furniture".
materials.csv is the wizard-produced bridge from a modifier name to a role,
and it is deliberately closed -- a modifier with no row, or a row with no
role, stops the run rather than falling back to a guess (R-P... "no
client-derived default, no silent default").

`glazing_interior` is its own role, disjoint from `EXTERIOR_GLAZING_ROLES`,
so that glass inside the envelope (partitions, internal windows, glazed
doors) can never be picked up by a later task's `"glazing" in role` check
and get a frame, a reveal, a blind or a window group it should not have.
"""
from __future__ import annotations

import csv
import io
import pathlib
from dataclasses import dataclass

from ..scene.input_file import InputFileError, read_input_bytes
from .config import ProjectError

#: The twelve roles a modifier may carry (spec 5.2). `glazing_diffusing`
#: is exterior glass that scatters light (a Radiance `trans` with no
#: see-through part): it gets no blind and is never vision glazing.
ROLES: frozenset[str] = frozenset({
    "glazing", "glazing_curtain_wall", "glazing_diffusing", "glazing_interior",
    "wall", "ceiling", "floor", "fixed_shading",
    "context", "ground", "furniture", "ignore",
})

#: Roles that make a real opening in the envelope -- the ones the frame,
#: reveal, blind and window-group logic (T3.9) may act on. `glazing_interior`
#: is glass but never an opening, so it must never appear here.
EXTERIOR_GLAZING_ROLES: frozenset[str] = frozenset(
    {"glazing", "glazing_curtain_wall", "glazing_diffusing"}
)

assert EXTERIOR_GLAZING_ROLES <= ROLES
assert "glazing_interior" not in EXTERIOR_GLAZING_ROLES

#: Columns materials.csv must have, in this exact order (spec 5.2: "exactly").
_HEADER = ("modifier", "primitive_type", "role", "note")

#: The optional last columns, in this order: `view` (Quality Views), `vlt`
#: (the calculator: visible light transmittance in %, from the product data).
_OPTIONAL = ("view", "vlt")
#: Roles that may carry a `vlt`: vision glazing to the outside.
VLT_ROLES: frozenset[str] = frozenset({"glazing", "glazing_curtain_wall"})

#: Quality Views spec 5.2: what a surface is to a view ray. Empty means
#: "use the role's default" (decided in build.view_model).
VIEW_VALUES: frozenset[str] = frozenset({"clear", "blocks", "nature", "art", "landmark"})
#: The three tags that satisfy Type 2 on sight ("nature, urban landmarks, or art").
TAG_VIEWS: frozenset[str] = frozenset({"nature", "art", "landmark"})
#: The roles a tag may sit on: things seen outside the glass, or a wall that
#: carries art. Never glass, a floor, a ceiling or furniture.
TAG_ROLES: frozenset[str] = frozenset({"context", "ground", "fixed_shading", "wall"})


@dataclass(frozen=True)
class MaterialRow:
    modifier: str
    primitive_type: str
    role: str
    note: str
    view: str = ""
    vlt: float | None = None


def _fail(detail: str) -> None:
    raise ProjectError(f"materials.csv: {detail}") from None


def _fail_at(line_no: int, detail: str) -> None:
    raise ProjectError(f"materials.csv line {line_no}: {detail}") from None


def read_csv_text(path: pathlib.Path, fail, show=repr) -> str:
    """Read a wizard CSV as UTF-8 text, or refuse through ``fail(detail)``.

    Shared by materials.csv and spaces.csv. ``utf-8-sig`` strips a UTF-8
    BOM, so a BOM-prefixed file does not read as if its first column were
    named '\ufeffmodifier'. Two things Excel does on a "Save as CSV" are
    refused by name here rather than surfacing as a bare
    ``UnicodeDecodeError`` or a misleading "missing column": saving in the
    Windows code page (cp1252) instead of UTF-8, and using ';' as the
    delimiter (locales with a decimal comma). ``show`` formats the header
    line a ';' refusal quotes (``repr`` by default; the points file passes a
    function that shows only the count of ';', none of the line's text).
    """
    try:
        data = read_input_bytes(path)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    except InputFileError as exc:
        fail(str(exc))
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        line_no = data[:exc.start].count(b"\n") + 1
        fail(
            f"{path} is not UTF-8 text (byte 0x{data[exc.start]:02x} at line {line_no}); "
            "it was probably saved in a Windows code page -- save it again as "
            "'CSV UTF-8 (Comma delimited)'"
        )
    header_line = text.split("\n", 1)[0]
    if ";" in header_line:
        fail(
            f"{path}: the header line uses ';' as the delimiter ({show(header_line.strip())}); "
            "the columns must be comma-separated -- save it again as "
            "'CSV UTF-8 (Comma delimited)'"
        )
    return text


def _check_header(fieldnames) -> tuple[str, ...]:
    """Validate the header and return it: the four columns, then optionally
    `view`, then optionally `vlt`."""
    normalized = tuple(name.strip() for name in (fieldnames or ()))
    allowed = {_HEADER, _HEADER + ("view",), _HEADER + ("vlt",), _HEADER + ("view", "vlt")}
    if normalized in allowed:
        return normalized
    shapes = f"{','.join(_HEADER)} (optionally followed by ',view' and then ',vlt')"
    missing = [c for c in _HEADER if c not in normalized]
    extra = [c for c in normalized if c not in _HEADER + _OPTIONAL]
    if missing:
        _fail(f"missing column {missing[0]!r}; columns must be exactly {shapes}, got {','.join(normalized)}")
    if extra:
        _fail(f"unexpected column {extra[0]!r}; columns must be exactly {shapes}, got {','.join(normalized)}")
    _fail(f"columns must be exactly {shapes}, got {','.join(normalized)}")


def _load_rows(path: str | pathlib.Path) -> tuple[MaterialRow, ...]:
    """Read, validate and return every row of materials.csv, once.

    Both public entry points call this; there is exactly one place that
    parses and validates the file.

    Line numbers in every refusal are FILE line numbers (header = line 1),
    computed from ``csv.reader``'s own ``line_num`` -- the count of physical
    source lines consumed so far -- rather than from counting *records*.
    A blank line still consumes one physical line, and a quoted field with
    an embedded newline spans several; either would silently shift every
    later line number if records were counted instead. A record is reported
    at the physical line it STARTS on, not the line ``line_num`` points to
    when the record finishes (its last line, for a multi-line record).
    """
    path = pathlib.Path(path)
    f = io.StringIO(read_csv_text(path, _fail), newline="")

    with f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            _fail(f"empty file; expected header {','.join(_HEADER)}")
        header = _check_header(header)

        result: list[MaterialRow] = []
        first_seen_at: dict[str, int] = {}
        last_line = reader.line_num
        while True:
            try:
                raw_row = next(reader)
            except StopIteration:
                break
            line_no = last_line + 1
            last_line = reader.line_num
            if raw_row == []:
                # A genuinely blank physical line: not a data row.
                continue

            if len(raw_row) != len(header):
                _fail_at(
                    line_no,
                    f"row has {len(raw_row)} field(s), expected {len(header)} "
                    f"({','.join(header)}): {raw_row!r}",
                )

            cells = [cell.strip() for cell in raw_row]
            modifier, primitive_type, role, note = cells[:4]
            extra = dict(zip(header[4:], cells[4:]))
            view = extra.get("view", "")
            vlt_text = extra.get("vlt", "")
            vlt = None
            if vlt_text:
                try:
                    vlt = float(vlt_text)
                except ValueError:
                    _fail_at(line_no, f"modifier {modifier!r}: vlt must be a number (percent), got {vlt_text!r}")
                if not 0 <= vlt <= 100:
                    _fail_at(line_no, f"modifier {modifier!r}: vlt must be from 0 to 100 (percent), got {vlt_text}")
                if role not in VLT_ROLES:
                    _fail_at(line_no, f"modifier {modifier!r}: vlt is allowed only on roles {sorted(VLT_ROLES)}, "
                                      f"not {role!r}")

            if not modifier:
                _fail_at(line_no, "modifier is blank; every row needs a modifier name")
            if not role:
                _fail_at(
                    line_no,
                    f"modifier {modifier!r} has no role. Every modifier needs one; "
                    "use 'ignore' if it is deliberately excluded.",
                )
            if role not in ROLES:
                _fail_at(
                    line_no,
                    f"modifier {modifier!r} has unknown role {role!r}; "
                    f"expected one of {sorted(ROLES)}",
                )
            if view and view not in VIEW_VALUES:
                _fail_at(
                    line_no,
                    f"modifier {modifier!r} has unknown view {view!r}; expected one of "
                    f"{sorted(VIEW_VALUES)} or empty",
                )
            if role == "glazing_diffusing" and view not in ("", "blocks"):
                _fail_at(
                    line_no,
                    f"modifier {modifier!r}: role 'glazing_diffusing' is never vision "
                    f"glazing, so its view must be empty or 'blocks', got {view!r}",
                )
            if role == "glazing_interior" and primitive_type == "trans" and view not in ("", "blocks"):
                _fail_at(
                    line_no,
                    f"modifier {modifier!r}: a 'trans' with role 'glazing_interior' is frosted "
                    f"glass, never vision glazing, so its view must be empty or 'blocks', got {view!r}",
                )
            if view in TAG_VIEWS and role not in TAG_ROLES:
                _fail_at(
                    line_no,
                    f"modifier {modifier!r}: view {view!r} is allowed only on roles "
                    f"{sorted(TAG_ROLES)}, not {role!r}",
                )
            if modifier in first_seen_at:
                _fail(
                    f"modifier {modifier!r} appears twice, at line {first_seen_at[modifier]} "
                    f"and line {line_no}"
                )
            first_seen_at[modifier] = line_no

            result.append(MaterialRow(
                modifier=modifier, primitive_type=primitive_type, role=role, note=note, view=view, vlt=vlt,
            ))

    if not result:
        _fail("no modifiers")

    return tuple(result)


def load_materials(path: str | pathlib.Path) -> dict[str, str]:
    """Return modifier -> role for every row of materials.csv, or refuse."""
    return {row.modifier: row.role for row in _load_rows(path)}


def declared_types(path: str | pathlib.Path) -> dict[str, str]:
    """Return modifier -> primitive_type for every row whose type cell is not blank."""
    return {row.modifier: row.primitive_type for row in _load_rows(path) if row.primitive_type}


def rows(path: str | pathlib.Path) -> list[MaterialRow]:
    """Return every row of materials.csv, notes included, for the report."""
    return list(_load_rows(path))


def view_overrides(path: str | pathlib.Path) -> dict[str, str]:
    """Return modifier -> view for every row whose `view` cell is not empty."""
    return {row.modifier: row.view for row in _load_rows(path) if row.view}


def vlt_values(path: str | pathlib.Path) -> dict[str, float]:
    """Return modifier -> VLT (percent) for every row whose `vlt` cell is not empty."""
    return {row.modifier: row.vlt for row in _load_rows(path) if row.vlt is not None}


def vlt_warnings(path: str | pathlib.Path) -> list[str]:
    """One warning per modifier whose `vlt` is above 0 and at most 1: VLT is
    a percentage, so 0.62 is probably meant as 62."""
    return [
        f"materials.csv: modifier {modifier!r} has vlt {vlt:g}; vlt is a percentage, so {vlt:g} is under 1 %: "
        f"did you mean {vlt * 100:g}?"
        for modifier, vlt in vlt_values(path).items() if 0 < vlt <= 1
    ]
