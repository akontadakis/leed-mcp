# leed_daylight/engine/sky.py
"""The annual sky matrix, the annual sun matrix, and a clear sky (layer 4).

Four Radiance commands, all going through :class:`~.radiance.Radiance` so
every one of them lands in the run log:

* :func:`sky_matrix` -- ``gendaymtx``, the Reinhart annual sky patches sDA
  and ASE integrate against.
* :func:`sun_matrix` -- ``gendaymtx -D``, one unit-radiance sun per sun-up
  hour, feeding ASE and the blind trigger.
* :func:`clear_sky` -- ``gensky ... +s``, a CIE clear sky; Option 2 now
  uses :func:`weather_sky` instead.
* :func:`weather_sky` -- ``gendaylit -W``, Option 2's sky from the weather
  file's own irradiance at a checked hour.

The building's north rotation is applied exactly ONCE, in this module,
by ``gendaymtx -r`` (for the sky and the sun) and by ``xform -rz`` on
gensky's own output (for the clear sky). The geometry is never rotated a
second time anywhere in this step -- that silent double-rotation is the
classic error the north-rotation number in every report has to survive.

Both matrix functions pass an explicit achromatic sky colour
(``-c 1 1 1``): gendaymtx's own default sky colour is bluish and would
tint every lux value that comes out of it. The sun and sky matrices are
fingerprinted on the ``.wea`` bytes, ``mf`` and the rotation (plus that
colour), so a stopped run resumes instead of re-running ``gendaymtx``.
Fingerprinting ``clear_sky`` is not required: it is cheap and its output
is a single small text file consumed once per Option 2 hour.
"""
from __future__ import annotations

import calendar
import json
import math
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass

import numpy as np

from ..project.config import Site
from .fingerprint import delete_sidecar, fingerprint_of, is_current, write_sidecar
from .radiance import Radiance

#: gendaymtx's own default sky colour is bluish (a warmer, higher-blue
#: triplet than 1:1:1); every call in this module overrides it so lux
#: values are never silently tinted. Held here once, and folded into every
#: fingerprint's parameters, so a colour change is itself a cache miss.
_SKY_COLOR: tuple[str, str, str] = ("1", "1", "1")

#: Any non-leap year, used only to validate that a (month, day) pair
#: exists; LEED daylight analysis never depends on which year it is.
_CALENDAR_YEAR = 2025

#: A sun primitive's name is ``sun<minute>`` per gendaymtx's ``-D`` output
#: (its modifier is ``solar<minute>``, same suffix).
_SUN_NAME_RE = re.compile(r"^sun(\d+)$")


class SkyError(ValueError):
    """A sky/sun command's inputs are unusable, or its output is malformed."""


@dataclass(frozen=True)
class Suns:
    """One annual sun matrix, read back from its ``.suns.json`` sidecar.

    ``modifiers`` and ``hours`` line up with ``rgb`` row for row, in the
    order ``gendaymtx -M`` gave. ``rgb`` is the TRUE per-sun radiance
    (before :func:`sun_matrix` resets it to unit radiance in ``out``).
    """

    modifiers: tuple[str, ...]
    hours: np.ndarray
    rgb: np.ndarray


# --- shared validation and formatting ------------------------------------


def _validate_mf(mf: object) -> None:
    if isinstance(mf, bool) or not isinstance(mf, int) or mf < 1:
        raise SkyError(f"mf must be an int >= 1, got {mf!r}")


def _validate_rotation(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkyError(f"{label} must be a finite number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise SkyError(f"{label} must be finite, got {value!r}")
    return as_float


def _format_number(value: float) -> str:
    """``str(float(value))``, so ``30`` becomes ``30.0`` and ``0`` becomes
    ``0.0`` -- an omitted flag looks like a forgotten one in the run log."""
    return str(float(value))


def _require_wea(wea: pathlib.Path) -> None:
    """Refuse, before ``gendaymtx`` runs, a weather file that is not a
    ``.wea``. ``gendaymtx`` reads only the ``.wea`` format; an ``.epw``
    handed to it either fails or yields a sun matrix whose time stamps
    match no line of the file, which used to surface only afterwards as a
    time-stamp-convention error."""
    if wea.suffix.lower() != ".wea":
        raise SkyError(
            f"weather file {wea}: must be a .wea file, got "
            f"{wea.suffix or 'no suffix'!s}; gendaymtx reads only .wea -- "
            "convert an .epw first (e.g. with Radiance's epw2wea)"
        )


def _mod_path(out: pathlib.Path) -> pathlib.Path:
    return out.with_name(out.name + ".mod")


def _suns_json_path(out: pathlib.Path) -> pathlib.Path:
    return out.with_name(out.name + ".suns.json")


# --- sky_matrix -----------------------------------------------------------


def sky_matrix(
    rad: Radiance, wea: pathlib.Path, out: pathlib.Path,
    mf: int, north_rotation_deg: float,
) -> pathlib.Path:
    """``gendaymtx -m <mf> -r <rot> -of -c 1 1 1 <wea>`` -> ``out``.

    ``out`` holds the binary sky matrix (with its Radiance header);
    ``gendaymtx`` writes it directly via ``stdout_path``. Skipped (the
    binary is not re-run) when ``out`` already exists and is current
    against the fingerprint of ``wea``'s bytes, ``mf`` and the rotation.
    A weather file whose suffix is not ``.wea`` is refused first.
    """
    wea = pathlib.Path(wea)
    _require_wea(wea)
    out = pathlib.Path(out)
    _validate_mf(mf)
    rotation = _validate_rotation(north_rotation_deg, "north_rotation_deg")
    rot_arg = _format_number(rotation)

    fp = fingerprint_of(
        {"wea": wea},
        {
            "command": "sky_matrix",
            "radiance_version": rad.version,
            "mf": mf,
            "rotation": rotation,
            "color": list(_SKY_COLOR),
        },
    )

    if out.exists() and is_current(out, fp):
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    delete_sidecar(out)  # M1: before the rebuild, not after
    argv = [
        "gendaymtx", "-m", str(mf), "-r", rot_arg, "-of",
        "-c", *_SKY_COLOR, str(wea),
    ]
    rad.run(argv, stdout_path=out)
    write_sidecar(out, fp)
    return out


# --- sun_matrix -------------------------------------------------------------


def _parse_radiance_primitives(text: str, *, source: pathlib.Path) -> list[dict]:
    """Tolerantly parse whitespace-separated Radiance primitives.

    Each primitive is ``modifier type name``, then three (count, that many
    args) groups: strings, integers, reals. Exact line breaks are never
    relied on. Any mismatch between a declared count and the tokens that
    follow -- a non-integer count, or fewer tokens than declared -- is a
    named :class:`SkyError` naming ``source``.
    """
    tokens = text.split()
    n = len(tokens)
    i = 0
    primitives: list[dict] = []
    while i < n:
        if i + 3 > n:
            raise SkyError(
                f"malformed Radiance primitive in {source}: header "
                "incomplete (modifier/type/name expected)"
            )
        modifier, ptype, name = tokens[i], tokens[i + 1], tokens[i + 2]
        i += 3
        arg_groups: list[list[str]] = []
        for group_name in ("string", "integer", "real"):
            if i >= n:
                raise SkyError(
                    f"malformed Radiance primitive {name!r} in {source}: "
                    f"missing the {group_name} argument count"
                )
            count_token = tokens[i]
            try:
                count = int(count_token)
            except ValueError:
                raise SkyError(
                    f"malformed Radiance primitive {name!r} in {source}: "
                    f"{group_name} argument count {count_token!r} is not "
                    "an integer -- the counts disagree"
                ) from None
            i += 1
            if count < 0 or i + count > n:
                raise SkyError(
                    f"malformed Radiance primitive {name!r} in {source}: "
                    f"declared {count} {group_name} argument(s) but they "
                    "are not all present -- the counts disagree"
                )
            arg_groups.append(tokens[i:i + count])
            i += count
        s_args, i_args, r_args = arg_groups
        primitives.append({
            "modifier": modifier, "type": ptype, "name": name,
            "s_args": s_args, "i_args": i_args, "r_args": r_args,
        })
    return primitives


def _to_float(token: str, *, name: str, source: pathlib.Path) -> float:
    try:
        return float(token)
    except ValueError:
        raise SkyError(
            f"real argument {token!r} of {name!r} in {source} is not a "
            "number"
        ) from None


def _require_no_string_or_int_args(prim: dict, *, source: pathlib.Path) -> None:
    """Radiance ``light``/``source`` primitives take zero string and zero
    integer arguments. Refusing here, before the real-argument count is
    even looked at, means a declared-count overrun can never silently
    shift the parse into the next primitive's tokens."""
    if prim["s_args"] or prim["i_args"]:
        raise SkyError(
            f"{prim['type']} {prim['name']!r} in {source} has "
            f"{len(prim['s_args'])} string and {len(prim['i_args'])} "
            "integer argument(s); a Radiance light/source primitive takes "
            "none of either -- the counts disagree"
        )


def _index_suns(
    primitives: list[dict], *, source: pathlib.Path,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, str]]:
    """``{light name: (R, G, B)}`` and ``{light modifier: source name}``."""
    light_by_name: dict[str, tuple[float, float, float]] = {}
    source_by_modifier: dict[str, str] = {}
    for prim in primitives:
        if prim["type"] == "light":
            _require_no_string_or_int_args(prim, source=source)
            if len(prim["r_args"]) != 3:
                raise SkyError(
                    f"light {prim['name']!r} in {source} has "
                    f"{len(prim['r_args'])} real argument(s), expected 3 "
                    "(R G B) -- the counts disagree"
                )
            r, g, b = (
                _to_float(v, name=prim["name"], source=source)
                for v in prim["r_args"]
            )
            light_by_name[prim["name"]] = (r, g, b)
        elif prim["type"] == "source":
            _require_no_string_or_int_args(prim, source=source)
            if len(prim["r_args"]) != 4:
                raise SkyError(
                    f"source {prim['name']!r} in {source} has "
                    f"{len(prim['r_args'])} real argument(s), expected 4 "
                    "(dx dy dz angle) -- the counts disagree"
                )
            source_by_modifier[prim["modifier"]] = prim["name"]
    return light_by_name, source_by_modifier


def _day_of_year(month: int, day: int) -> int:
    """1-based day of the year, in :data:`_CALENDAR_YEAR` (a non-leap
    year), so January 1st is day 1."""
    cumulative = 0
    for m in range(1, month):
        cumulative += calendar.monthrange(_CALENDAR_YEAR, m)[1]
    return cumulative + day


def _wea_minute_index(wea: pathlib.Path) -> dict[int, int]:
    """``{minute-of-year: 0-based line index}`` for every DATA line in
    ``wea`` (I3): gendaymtx names a sun ``sun<minute>``, "numbered from the
    minute, from midnight, January 1st" (gendaymtx(1)); this computes that
    SAME minute-of-year for each of the weather file's own time-step
    lines, so a sun's name can be matched back to the COLUMN its own line
    produced in the sky/sun matrix (that column is the line's plain
    position among the data lines, not a value derived from its own
    minute -- a ``.wea`` need not have one line per hour of the year).

    A header line (``place``, ``latitude``, ...) never has three leading
    numeric tokens and is skipped without being counted; a data line whose
    ``month``/``day`` do not form a real date in :data:`_CALENDAR_YEAR` is
    skipped too (never silently counted as a line the matrix does not
    actually have).
    """
    text = wea.read_text(encoding="utf-8", errors="replace")
    index: dict[int, int] = {}
    line_no = 0
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 3:
            continue
        try:
            month = int(parts[0])
            day = int(parts[1])
            hour = float(parts[2])
        except ValueError:
            continue
        if not (1 <= month <= 12):
            continue
        days_in_month = calendar.monthrange(_CALENDAR_YEAR, month)[1]
        if not (1 <= day <= days_in_month):
            continue
        minute = round((_day_of_year(month, day) - 1) * 1440 + hour * 60)
        index[minute] = line_no
        line_no += 1
    return index


def _sun_records(
    modifiers: list[str],
    light_by_name: dict[str, tuple[float, float, float]],
    source_by_modifier: dict[str, str],
    *, source: pathlib.Path, wea: pathlib.Path, minute_index: dict[int, int],
) -> list[dict]:
    records: list[dict] = []
    for modifier in modifiers:
        light_rgb = light_by_name.get(modifier)
        if light_rgb is None:
            raise SkyError(
                f"modifier {modifier!r} listed in the sun modifier list "
                f"has no matching light primitive in {source}"
            )
        sun_name = source_by_modifier.get(modifier)
        if sun_name is None:
            raise SkyError(
                f"light {modifier!r} in {source} has no source primitive "
                "referencing it"
            )
        match = _SUN_NAME_RE.match(sun_name)
        if match is None:
            raise SkyError(
                f"sun name {sun_name!r} in {source} does not encode an "
                "integer minute"
            )
        minute = int(match.group(1))
        # I3: `minute` is the minute-of-year gendaymtx named this sun
        # with; the matrix COLUMN it belongs to is whichever of the
        # weather file's own data lines has that same minute-of-year --
        # never `minute // 60`, which silently assumes every line is a
        # whole-hour-of-day stamp starting at minute 0 of day 1.
        hour = minute_index.get(minute)
        if hour is None:
            raise SkyError(
                f"sun {sun_name!r} (modifier {modifier!r}) names minute "
                f"{minute}, which matches no data line in weather file "
                f"{wea}; the file's time-stamp convention could not be "
                "matched"
            )
        records.append({"modifier": modifier, "hour": hour, "rgb": list(light_rgb)})
    return records


def _unit_radiance_text(primitives: list[dict]) -> str:
    """``primitives`` re-serialised with every ``light`` RGB set to 1 1 1."""
    lines: list[str] = []
    for prim in primitives:
        r_args = ["1", "1", "1"] if prim["type"] == "light" else prim["r_args"]
        lines.append(f"{prim['modifier']} {prim['type']} {prim['name']}")
        lines.append(_count_line(prim["s_args"]))
        lines.append(_count_line(prim["i_args"]))
        lines.append(_count_line(r_args))
        lines.append("")
    return "\n".join(lines) + "\n"


def _count_line(args: list[str]) -> str:
    return " ".join([str(len(args)), *args]) if args else str(len(args))


def sun_matrix(
    rad: Radiance, wea: pathlib.Path, out: pathlib.Path,
    mf: int, north_rotation_deg: float,
) -> pathlib.Path:
    """``gendaymtx -n -D <suns> -M <modifiers> -r <rot> -d -m <mf> -c 1 1 1
    <wea>``: one unit-radiance sun per sun-up hour.

    Writes three files and returns ``out``: ``out`` itself (the parsed
    suns with every light reset to unit radiance ``1 1 1``), ``out.mod``
    (the modifier names, one per line, in gendaymtx's own order) and
    ``out.suns.json`` (the TRUE radiance per sun, before that reset, read
    back by :func:`read_suns`). Skipped only when all three already exist
    and ``out`` is current against the fingerprint of ``wea``'s bytes,
    ``mf`` and the rotation. A weather file whose suffix is not ``.wea``
    is refused first.
    """
    wea = pathlib.Path(wea)
    _require_wea(wea)
    out = pathlib.Path(out)
    _validate_mf(mf)
    rotation = _validate_rotation(north_rotation_deg, "north_rotation_deg")
    rot_arg = _format_number(rotation)

    fp = fingerprint_of(
        {"wea": wea},
        {
            "command": "sun_matrix",
            "radiance_version": rad.version,
            "mf": mf,
            "rotation": rotation,
            "color": list(_SKY_COLOR),
        },
    )

    mod_path = _mod_path(out)
    json_path = _suns_json_path(out)
    if (
        out.exists() and mod_path.exists() and json_path.exists()
        and is_current(out, fp)
    ):
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    delete_sidecar(out)  # M1: before the rebuild, not after
    tmp_suns_fd, tmp_suns_name = tempfile.mkstemp(
        dir=out.parent, prefix=f".{out.name}.suns.", suffix=".tmp",
    )
    tmp_mod_fd, tmp_mod_name = tempfile.mkstemp(
        dir=out.parent, prefix=f".{out.name}.mod.", suffix=".tmp",
    )
    os.close(tmp_suns_fd)
    os.close(tmp_mod_fd)
    tmp_suns_path = pathlib.Path(tmp_suns_name)
    tmp_mod_path = pathlib.Path(tmp_mod_name)
    try:
        argv = [
            "gendaymtx", "-n", "-D", str(tmp_suns_path),
            "-M", str(tmp_mod_path), "-r", rot_arg, "-d", "-m", str(mf),
            "-c", *_SKY_COLOR, str(wea),
        ]
        rad.run(argv)

        modifiers_text = tmp_mod_path.read_text(encoding="utf-8")
        modifiers = [
            line.strip() for line in modifiers_text.splitlines() if line.strip()
        ]
        suns_text = tmp_suns_path.read_text(encoding="utf-8")
        primitives = _parse_radiance_primitives(suns_text, source=tmp_suns_path)
        light_by_name, source_by_modifier = _index_suns(
            primitives, source=tmp_suns_path,
        )
        minute_index = _wea_minute_index(wea)
        records = _sun_records(
            modifiers, light_by_name, source_by_modifier,
            source=tmp_suns_path, wea=wea, minute_index=minute_index,
        )

        out.write_text(_unit_radiance_text(primitives), encoding="utf-8", newline="\n")
        mod_path.write_text(
            "".join(f"{m}\n" for m in modifiers), encoding="utf-8", newline="\n",
        )
        json_path.write_text(json.dumps(records), encoding="utf-8", newline="\n")
    finally:
        tmp_suns_path.unlink(missing_ok=True)
        tmp_mod_path.unlink(missing_ok=True)

    write_sidecar(out, fp)
    return out


def _require_suns_record_shape(
    record: object, index: int, *, json_path: pathlib.Path,
) -> tuple[str, int, list[float]]:
    """Validate one ``.suns.json`` record and return
    ``(modifier, hour, rgb)``, never letting a malformed shape reach a raw
    ``KeyError``/``TypeError``/numpy ``ValueError`` -- every mismatch is a
    named :class:`SkyError` naming the file, the record index, and what
    was wrong."""
    if not isinstance(record, dict):
        raise SkyError(
            f"suns json {json_path} record {index} must be an object, "
            f"got {type(record).__name__}: {record!r}"
        )
    for key in ("modifier", "hour", "rgb"):
        if key not in record:
            raise SkyError(
                f"suns json {json_path} record {index} is missing {key!r}: "
                f"{record!r}"
            )

    modifier = record["modifier"]
    if not isinstance(modifier, str):
        raise SkyError(
            f"suns json {json_path} record {index} 'modifier' must be a "
            f"string, got {type(modifier).__name__}: {modifier!r}"
        )

    hour = record["hour"]
    if isinstance(hour, bool) or not isinstance(hour, int) or hour < 0:
        raise SkyError(
            f"suns json {json_path} record {index} 'hour' must be a "
            f"non-negative int, got {hour!r}"
        )

    rgb = record["rgb"]
    if not isinstance(rgb, list) or len(rgb) != 3:
        raise SkyError(
            f"suns json {json_path} record {index} 'rgb' must be a list "
            f"of three numbers, got {rgb!r}"
        )
    rgb_values: list[float] = []
    for component in rgb:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise SkyError(
                f"suns json {json_path} record {index} 'rgb' component "
                f"{component!r} is not a number"
            )
        value = float(component)
        if not math.isfinite(value):
            raise SkyError(
                f"suns json {json_path} record {index} 'rgb' component "
                f"{component!r} is not finite"
            )
        rgb_values.append(value)

    return modifier, hour, rgb_values


def read_suns(path: pathlib.Path) -> Suns:
    """Read the ``<path>.suns.json`` sidecar :func:`sun_matrix` wrote
    beside ``path`` (``path`` is the ``out`` :func:`sun_matrix` returned).

    Every record is validated before it reaches a numpy array: a document
    that is not a list, a record that is not an object, a missing
    ``modifier``/``hour``/``rgb``, an ``rgb`` that is not exactly three
    finite numbers, or an ``hour`` that is not a non-negative int is a
    named :class:`SkyError` naming the file and the record.
    """
    path = pathlib.Path(path)
    json_path = _suns_json_path(path)
    try:
        raw = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkyError(f"suns json {json_path} could not be read: {exc}") from exc
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SkyError(f"suns json {json_path} is not valid JSON: {exc}") from exc

    if not isinstance(records, list):
        raise SkyError(
            f"suns json {json_path} must be a list of records, got "
            f"{type(records).__name__}"
        )

    modifiers: list[str] = []
    hours: list[int] = []
    rgb_rows: list[list[float]] = []
    for index, record in enumerate(records):
        modifier, hour, rgb_values = _require_suns_record_shape(
            record, index, json_path=json_path,
        )
        modifiers.append(modifier)
        hours.append(hour)
        rgb_rows.append(rgb_values)

    return Suns(
        modifiers=tuple(modifiers),
        hours=np.array(hours, dtype=int),
        rgb=np.array(rgb_rows, dtype=np.float64).reshape(-1, 3),
    )


# --- clear_sky --------------------------------------------------------------


def _validate_month(month: object) -> None:
    if isinstance(month, bool) or not isinstance(month, int) or not (1 <= month <= 12):
        raise SkyError(f"month must be an int between 1 and 12, got {month!r}")


def _validate_day(month: int, day: object) -> None:
    if isinstance(day, bool) or not isinstance(day, int):
        raise SkyError(f"day must be an int, got {day!r}")
    days_in_month = calendar.monthrange(_CALENDAR_YEAR, month)[1]
    if not (1 <= day <= days_in_month):
        raise SkyError(
            f"day {day!r} does not exist in month {month!r} "
            f"(1..{days_in_month})"
        )


def _validate_hour(hour: object) -> float:
    if isinstance(hour, bool) or not isinstance(hour, (int, float)):
        raise SkyError(f"hour must be a number, got {hour!r}")
    value = float(hour)
    if not math.isfinite(value) or not (0.0 <= value <= 24.0):
        raise SkyError(f"hour must be between 0 and 24, got {hour!r}")
    return value


def clear_sky(
    rad: Radiance, out: pathlib.Path, month: int, day: int, hour: float,
    site: Site, north_rotation_deg: float,
) -> pathlib.Path:
    """A clear sky at ``month``/``day``/``hour``, rotated by the north
    rotation.

    Two commands, both through :meth:`Radiance.run`: ``gensky <month>
    <day> <hour> +s -a <latitude_N> -o <-longitude_E> -m
    <meridian_radiance>`` (its stdout captured, never written to disk),
    then ``xform -rz <rot>`` with that description as ``stdin`` and
    ``out`` as ``stdout_path``. This rotates the sky description, never
    the geometry -- the geometry's own rotation happens nowhere in this
    module. ``north_rotation_deg`` is passed explicitly by the caller
    (normally ``project.site.north_rotation_deg``); this function never
    reads a ``Project``.

    ``gensky``'s ``-o`` is degrees WEST positive, so the east-positive
    ``site.longitude_E`` is negated; ``site.meridian_radiance`` is already
    in Radiance's own convention and passes through unchanged.
    """
    out = pathlib.Path(out)
    _validate_month(month)
    _validate_day(month, day)
    hour_value = _validate_hour(hour)
    rotation = _validate_rotation(north_rotation_deg, "north_rotation_deg")
    rot_arg = _format_number(rotation)

    gensky_argv = [
        "gensky", str(int(month)), str(int(day)), _format_number(hour_value),
        "+s",
        "-a", _format_number(site.latitude_N),
        "-o", _format_number(-site.longitude_E),
        "-m", _format_number(site.meridian_radiance),
    ]
    sky_description = rad.run(gensky_argv)

    out.parent.mkdir(parents=True, exist_ok=True)
    rad.run(["xform", "-rz", rot_arg], stdin=sky_description, stdout_path=out)
    return out


# --- weather_sky: Option 2's sky, from the weather file ------------------

#: gendaylit prints only the sun and the ``skyfunc`` brightness function;
#: the glowing sky and ground hemispheres that carry it are the caller's.
SKY_AND_GROUND_GLOW = """
skyfunc glow sky_glow
0
0
4 1 1 1 0

sky_glow source sky
0
0
4 0 0 1 180

skyfunc glow ground_glow
0
0
4 1 1 1 0

ground_glow source ground
0
0
4 0 0 -1 180
"""


def _validate_irradiance(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkyError(f"{label} must be a number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float) or as_float < 0.0:
        raise SkyError(f"{label} must be finite and not negative, got {value!r}")
    return as_float


def weather_sky(
    rad: Radiance, out: pathlib.Path, month: int, day: int, hour: float,
    dni: float, dhi: float, site: Site, north_rotation_deg: float,
) -> pathlib.Path:
    """Option 2's sky at ``month``/``day``/``hour`` (standard time) from
    the weather file's own direct normal and diffuse horizontal irradiance
    (W/m2): ``gendaylit ... -W <dni> <dhi>``, the sky and ground glow, then
    ``xform -rz`` -- the same single rotation :func:`clear_sky` applies.

    Refused: a zero diffuse irradiance (before gendaylit runs), and an
    "error sky". Radiance 6.1a's gendaylit, given a sun below the horizon,
    zero irradiance with the sun up, or a DNI above the solar constant,
    prints a warning to stderr, writes an all-zero sky with no sun and
    exits 0; every point would then read 0 lux. A real sky always carries ``solar source sun``, even with zero
    direct irradiance, so its absence is the test.
    """
    out = pathlib.Path(out)
    _validate_month(month)
    _validate_day(month, day)
    hour_value = _validate_hour(hour)
    dni_value = _validate_irradiance(dni, "dni")
    dhi_value = _validate_irradiance(dhi, "dhi")
    if dhi_value <= 0.0:
        raise SkyError(
            f"{month:02d}-{day:02d} at {hour_value:g}:00: the diffuse irradiance is 0 "
            "W/m2; gendaylit would write an all-zero error sky"
        )
    rotation = _validate_rotation(north_rotation_deg, "north_rotation_deg")

    argv = [
        "gendaylit", str(int(month)), str(int(day)), _format_number(hour_value),
        "-a", _format_number(site.latitude_N),
        "-o", _format_number(-site.longitude_E),
        "-m", _format_number(site.meridian_radiance),
        "-W", _format_number(dni_value), _format_number(dhi_value),
    ]
    description = rad.run(argv)
    if b"source sun" not in description:
        raise SkyError(
            f"{month:02d}-{day:02d} at {hour_value:g}:00: gendaylit wrote an error sky "
            f"(no sun; DNI {dni_value:g}, DHI {dhi_value:g} W/m2) -- the sun is below "
            "the horizon, the irradiance is zero, or the DNI exceeds the solar constant"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    rad.run(
        ["xform", "-rz", _format_number(rotation)],
        stdin=description + SKY_AND_GROUND_GLOW.encode("ascii"),
        stdout_path=out,
    )
    return out
