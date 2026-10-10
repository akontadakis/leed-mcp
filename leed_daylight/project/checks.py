# leed_daylight/project/checks.py
"""Cross-field and filesystem checks on a loaded ``Project`` (layer 2).

``config.py`` only checks that a key is present, has the right type and, for
a closed vocabulary, is one of the permitted words. It never opens a file it
is only pointed at and it never compares one field against another. That is
this module's job: does a referenced file actually exist, is a fraction
really a fraction, does one range make sense against another.

``check_project`` never raises by itself, for the same reason
``scene.validate.validate_scene`` never raises: one bad key must not hide
the other nine. ``load_project`` collects every error before deciding
whether to refuse.

This module must not import ``leed_daylight.metrics`` (project is layer 2,
metrics is layer 5; a layer may import only the layer below it -- R-P13).
The occupied-hours grammar is therefore duplicated here from
``metrics.occupancy.occupied_mask`` rather than imported, and
``test_the_hours_check_agrees_with_occupied_mask`` is what keeps the two
copies from drifting apart.
"""
from __future__ import annotations

import calendar
import datetime
import math
import os
import re
import typing
from collections.abc import Mapping

from ..constants import UNVERIFIED

if typing.TYPE_CHECKING:
    from .config import Project

#: Reinhart sky subdivisions this tool knows how to build (spec: sky.MF).
_MF_VALUES = frozenset({1, 2, 4, 6})

#: Mirrors ``metrics.occupancy.occupied_mask``'s own grammar exactly: two
#: digits, a colon, two digits, a hyphen, the same again.
_HOURS_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")

#: Mirrors ``metrics.occupancy.EARLIEST_DST_START_H``: under daylight
#: saving, an occupied window may not open before 03:00.
_EARLIEST_DST_START_H = 3

#: The band the tool has been checked against (spec 6.0 last row).
_LATITUDE_MIN_N = 37.0
_LATITUDE_MAX_N = 48.0

#: "grids should be no more than 2 feet (600 millimeters) square."
_MAX_GRID_SPACING_M = 0.60

#: A work plane at or above head height is a typo, not a design choice.
_MAX_WORK_PLANE_M = 2.0
_MAX_CLEARANCE_M = 2.0

#: A built skylight well must be deeper than this, in metres: the skylight
#: blind hangs this far below the glass (``build.blinds.BLIND_OFFSET_M``;
#: this layer may not import ``build``, and a test keeps the two equal).
_MIN_SKYLIGHT_WELL_M = 0.05

#: Atlas spec 8.1: a drawing's two reference points must lie at least this
#: far apart, on the drawing and in the model, to place it well.
_MIN_DRAWING_GAP_PX = 10.0
_MIN_DRAWING_GAP_M = 0.5

#: A standard meridian more than two time zones from the site is suspect.
_MERIDIAN_TOLERANCE_DEG = 30.0
#: Half a time zone: a meridian within this of 0 has no meaningful sign.
_HALF_ZONE_DEG = 7.5


def _hours_is_valid(hours: str) -> bool:
    """Apply the SAME rule ``metrics.occupancy.occupied_mask`` applies.

    Whole hours only (minutes ``00``), running forwards within one day,
    ``0 <= start < end <= 24``.
    """
    match = _HOURS_RE.match(hours)
    if not match:
        return False
    start_h, start_m, end_h, end_m = (int(g) for g in match.groups())
    if (start_m, end_m) != (0, 0):
        return False
    return 0 <= start_h < end_h <= 24


def _check_meridian(
    longitude_E: float, meridian_radiance: float, errors: list[str], warnings: list[str],
) -> None:
    """Compare the time-zone meridian against the site's own longitude.

    ``site.longitude_E`` is east positive; ``site.meridian_radiance`` follows
    Radiance (gendaymtx/gensky) and is WEST positive, so a consistent pair
    has ``meridian_radiance`` close to ``-longitude_E``. A standard time
    zone is rarely more than two zones (30 degrees) from the site. Far off
    AND of opposite sign, both clear of the prime-meridian zone, is the
    signature of a sign-convention mix (the east-positive zone written as
    is): an error. Far off otherwise is only unusual: a warning.
    """
    site_west = -longitude_E
    # Angular difference, so a site near the 180th meridian is not flagged
    # against a meridian written on the other side of it.
    apart = abs((site_west - meridian_radiance + 180.0) % 360.0 - 180.0)
    if apart <= _MERIDIAN_TOLERANCE_DEG:
        return
    detail = (
        f"site.meridian_radiance is {meridian_radiance!r} but site.longitude_E is "
        f"{longitude_E!r}; they are more than {_MERIDIAN_TOLERANCE_DEG:g} degrees apart. "
        "site.meridian_radiance is west-positive, as Radiance expects (a time zone "
        "15 degrees EAST of Greenwich, UTC+1, is -15), while site.longitude_E is "
        f"east-positive; for this site a meridian near {site_west:g} is expected"
    )
    mixed = (
        site_west * meridian_radiance < 0
        and abs(site_west) > _HALF_ZONE_DEG
        and abs(meridian_radiance) > _HALF_ZONE_DEG
    )
    if mixed:
        errors.append(detail + " -- the signs suggest the east-positive value was written")
    else:
        warnings.append(detail)


#: The six header lines gendaymtx reads before the data (``epw2wea``'s own
#: output), in order.
_WEA_HEADER_KEYS = (
    "place", "latitude", "longitude", "time_zone", "site_elevation",
    "weather_data_file_units",
)
#: One data line per hour of a non-leap year (``metrics.occupancy``'s mask).
_WEA_HOURS = 8760
#: The same non-leap calendar year ``engine.sky`` dates a data line in.
_WEA_YEAR = 2023
#: Days in the months before month m (index m, 1-based), in ``_WEA_YEAR``.
_WEA_DAYS_BEFORE_MONTH = [0, 0]
for _m in range(1, 12):
    _WEA_DAYS_BEFORE_MONTH.append(
        _WEA_DAYS_BEFORE_MONTH[-1] + calendar.monthrange(_WEA_YEAR, _m)[1]
    )
del _m
#: Days in each month (index m, 1-based), in ``_WEA_YEAR``.
_WEA_DAYS_IN_MONTH = [0] + [calendar.monthrange(_WEA_YEAR, m)[1] for m in range(1, 13)]
#: A weather file more than this far from project.yaml's site is suspect.
_WEA_SITE_TOLERANCE_DEG = 1.0


def _wea_hour_of_year(month: int, day: int, hour: float) -> int:
    """0-based hour of the year the interval stamped ``hour`` belongs to.

    ``ceil(hour) - 1`` puts 0.5 (and hour-ending 1.0) in hour 0 of the day.
    """
    day_of_year = _WEA_DAYS_BEFORE_MONTH[month] + day
    return (day_of_year - 1) * 24 + math.ceil(hour) - 1


def _check_weather(project: "Project", errors: list[str], warnings: list[str]) -> None:
    """Refuse a weather file the annual sky and sun mapping cannot use.

    ``engine.sky`` runs ``gendaymtx`` on the file and then matches every
    sun it names back to a data line by minute of the year; an ``.epw``
    (comma-separated, no ``.wea`` header) fails only there, after the sky
    matrix has already been computed. The occupancy mask and every metric
    assume one line per hour of the year, starting 1 January, and the
    minute-of-year a line maps to is taken from its own hour stamp -- so an
    hour-ending integer stamp (1..24, as some converters write) moves every
    sun 30 minutes from the middle of the hour it stands for.

    Data lines are recognised as ``engine.sky`` recognises them (three
    leading numeric tokens: month, day, hour), then held to the stricter
    shape above. Never raises.
    """
    key = "weather.file"
    path = project.weather_file
    if path.suffix.lower() != ".wea":
        errors.append(
            f"{key} must be a Radiance .wea file, got {path.name!r}; convert an "
            "EnergyPlus .epw with Radiance's epw2wea (epw2wea in.epw out.wea) and "
            "point weather.file at the .wea"
        )
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        errors.append(f"{key}: cannot read {path}: {exc}")
        return

    header: dict[str, str] = {}
    data: list[tuple[int, int, int, float]] = []  # (file line, month, day, hour)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        if parts[0] in _WEA_HEADER_KEYS:
            header[parts[0]] = " ".join(parts[1:])
            continue
        try:
            month, day, hour = int(parts[0]), int(parts[1]), float(parts[2])
            [float(p) for p in parts[3:5]]
        except (ValueError, IndexError):
            errors.append(
                f"{key}: line {line_no} of {path.name} is neither a .wea header line "
                f"({', '.join(_WEA_HEADER_KEYS)}) nor a data line 'month day hour "
                f"direct diffuse': {raw.strip()[:60]!r}"
            )
            return
        if len(parts) < 5:
            errors.append(
                f"{key}: line {line_no} of {path.name} has {len(parts)} field(s); a data "
                f"line is 'month day hour direct diffuse': {raw.strip()[:60]!r}"
            )
            return
        if not (1 <= month <= 12 and 1 <= day <= _WEA_DAYS_IN_MONTH[month]):
            errors.append(
                f"{key}: line {line_no} of {path.name}: month {month} day {day} is not a "
                "date in a non-leap year (a leap-year file must drop 29 February)"
            )
            return
        data.append((line_no, month, day, hour))

    missing = [k for k in _WEA_HEADER_KEYS if k not in header]
    if missing:
        errors.append(
            f"{key}: {path.name} has no .wea header line(s) {', '.join(missing)}; "
            f"gendaymtx needs all six ({', '.join(_WEA_HEADER_KEYS)}) -- write the "
            "file with epw2wea"
        )
    if len(data) != _WEA_HOURS:
        errors.append(
            f"{key}: {path.name} has {len(data)} data line(s); expected exactly "
            f"{_WEA_HOURS}, one per hour of a non-leap year"
        )
    if data and (data[0][1], data[0][2]) != (1, 1):
        errors.append(
            f"{key}: {path.name}'s first data line (line {data[0][0]}) is month "
            f"{data[0][1]} day {data[0][2]}; the year must start on January 1"
        )
    off_centre = [d for d in data if abs(d[3] - math.floor(d[3]) - 0.5) > 1e-6]
    if off_centre:
        line_no, month, day, hour = off_centre[0]
        errors.append(
            f"{key}: {len(off_centre)} data line(s) of {path.name} are not stamped on "
            f"the half hour, first at line {line_no} (month {month} day {day} hour "
            f"{hour:g}); each hour must be stamped at its middle (0.5, 1.5 ... 23.5) -- "
            "an hour-ending integer stamp moves every sun 30 minutes"
        )
    for index, (line_no, month, day, hour) in enumerate(data):
        if _wea_hour_of_year(month, day, hour) != index:
            errors.append(
                f"{key}: {path.name} line {line_no} (month {month} day {day} hour "
                f"{hour:g}) is out of place; data must be chronological, one line per "
                "hour, with no gap or repeat"
            )
            break

    site = project.site
    mismatched = []
    try:
        wea_latitude = float(header["latitude"])
        if abs(wea_latitude - site.latitude_N) > _WEA_SITE_TOLERANCE_DEG:
            mismatched.append(f"latitude {wea_latitude:g} vs site.latitude_N {site.latitude_N!r}")
    except (KeyError, ValueError):
        pass
    try:
        wea_longitude = float(header["longitude"])
        apart = abs((wea_longitude + site.longitude_E + 180.0) % 360.0 - 180.0)
        if apart > _WEA_SITE_TOLERANCE_DEG:
            mismatched.append(
                f"longitude {wea_longitude:g} (west-positive, as in Radiance) vs "
                f"site.longitude_E {site.longitude_E!r} (east-positive, so "
                f"{-site.longitude_E:g} expected)"
            )
    except (KeyError, ValueError):
        pass
    if mismatched:
        warnings.append(
            f"{key}: {path.name}'s header is more than {_WEA_SITE_TOLERANCE_DEG:g} degree "
            f"from project.yaml's site: {'; '.join(mismatched)}; check it is the "
            "right weather file"
        )


def _check_file(errors: list[str], key: str, path) -> None:
    if not (path.is_file() and os.access(path, os.R_OK)):
        errors.append(f"{key}: file does not exist or is not readable: {path}")


def check_project(project: "Project") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every cross-field and filesystem check. Returns ``(errors, warnings)``.

    Never raises by itself. Every refusal names the key and the value it
    saw. A warning never becomes a refusal: it is only ever added to the
    ``warnings`` returned here, for ``load_project`` to carry onto
    ``Project.warnings``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- filesystem (spec 4.1 layer 2, spec 5.4) ---
    _check_file(errors, "weather.file", project.weather_file)
    if errors == []:  # the file exists and is readable
        _check_weather(project, errors, warnings)
    for i, path in enumerate(project.scene.files):
        _check_file(errors, f"scene.files[{i}]", path)
    _check_file(errors, "standard.credit_rules", project.credit_rules_path)
    if project.grid.points_file is not None:
        _check_file(errors, "grid.points_file", project.grid.points_file)

    # --- glazing.frame_share (spec 12 rule 3: no client-derived default) ---
    # Every key in this OPEN mapping, "default" included, must be stated: a
    # null value means the project named a role but gave it no fraction,
    # and there is no built-in default to fall back to.
    for role, value in project.glazing.frame_share.items():
        key = f"glazing.frame_share.{role}"
        if value is None:
            errors.append(f"{key} must be stated; there is no built-in default, got None")
        elif not 0 <= value < 1:
            errors.append(f"{key} must be 0 <= s < 1, got {value!r}")

    # --- glazing fractions ---
    dirt_factor = project.glazing.dirt_factor
    if isinstance(dirt_factor, Mapping):
        dirt_items = [(f"glazing.dirt_factor.{key}", value) for key, value in dirt_factor.items()]
    else:
        dirt_items = [("glazing.dirt_factor", dirt_factor)]
    for key, value in dirt_items:
        if not 0 < value <= 1:
            errors.append(
                f"{key} must be 0 < f <= 1 (LM-83-12 clause 2.2.9 is a "
                f"reduction, never a gain), got {value!r}"
            )
    well = project.glazing.skylight_well
    well_reflectance = project.glazing.skylight_well_reflectance
    if isinstance(well, float) and not well > _MIN_SKYLIGHT_WELL_M:
        errors.append(
            f"glazing.skylight_well must be 'modelled' or a depth above "
            f"{_MIN_SKYLIGHT_WELL_M:g} m (the skylight blind hangs "
            f"{_MIN_SKYLIGHT_WELL_M:g} m below the glass, inside the well), got {well!r}"
        )
    if well_reflectance is not None and not 0 <= well_reflectance <= 1:
        errors.append(
            f"glazing.skylight_well_reflectance must be 0 <= r <= 1, got {well_reflectance!r}"
        )
    if well == "modelled" and well_reflectance is not None:
        errors.append(
            "glazing.skylight_well_reflectance is given, but glazing.skylight_well is "
            "'modelled': the tool builds no well, so remove the reflectance"
        )
    if well is None and well_reflectance is not None:
        errors.append(
            "glazing.skylight_well_reflectance is given without glazing.skylight_well"
        )
    if isinstance(well, float) and well_reflectance is None:
        errors.append(
            "glazing.skylight_well_reflectance must be stated when glazing.skylight_well "
            f"is a depth ({well!r} m); there is no built-in default"
        )
    frame_reflectance = project.glazing.frame_reflectance
    if not 0 <= frame_reflectance <= 1:
        errors.append(
            f"glazing.frame_reflectance must be 0 <= r <= 1, got {frame_reflectance!r}"
        )
    reveal_reflectance = project.glazing.reveal_reflectance
    if not 0 <= reveal_reflectance <= 1:
        errors.append(
            f"glazing.reveal_reflectance must be 0 <= r <= 1, got {reveal_reflectance!r}"
        )
    reveal_depth_m = project.glazing.reveal_depth_m
    if reveal_depth_m < 0:
        errors.append(f"glazing.reveal_depth_m must be >= 0 (0 disables reveals), got {reveal_depth_m!r}")

    # --- reflectance.by_role (a reflectance above 1 creates light) ---
    for role, value in project.reflectance.by_role.items():
        if not 0 <= value <= 1:
            errors.append(f"reflectance.{role} must be 0 <= r <= 1, got {value!r}")

    # --- grid ---
    spacing_m = project.grid.spacing_m
    if not 0 < spacing_m <= _MAX_GRID_SPACING_M:
        errors.append(
            "grid.spacing_m must be greater than zero and no more than 2 feet "
            f"(600 millimeters) square: 0 < s <= {_MAX_GRID_SPACING_M}, got {spacing_m!r}"
        )
    work_plane_m = project.grid.work_plane_m
    if not 0 < work_plane_m < _MAX_WORK_PLANE_M:
        errors.append(
            f"grid.work_plane_m must be 0 < h < {_MAX_WORK_PLANE_M} (a work plane at "
            f"or above head height is a typo), got {work_plane_m!r}"
        )
    min_clearance_m = project.grid.min_clearance_m
    if not 0 <= min_clearance_m < _MAX_CLEARANCE_M:
        errors.append(
            f"grid.min_clearance_m must be 0 <= c < {_MAX_CLEARANCE_M}, got {min_clearance_m!r}"
        )

    # --- blinds ---
    diffuse = project.blinds.transmittance_diffuse
    if not 0 <= diffuse <= 1:
        errors.append(f"blinds.transmittance_diffuse must be 0 <= t <= 1, got {diffuse!r}")
    specular = project.blinds.transmittance_specular
    if not 0 <= specular <= 1:
        errors.append(f"blinds.transmittance_specular must be 0 <= t <= 1, got {specular!r}")
    if diffuse + specular > 1:
        errors.append(
            "blinds.transmittance_diffuse + blinds.transmittance_specular must be <= 1 "
            f"(a shade cannot transmit more than it receives), got {diffuse!r} + "
            f"{specular!r} = {diffuse + specular!r}"
        )
    room_side_reflectance = project.blinds.room_side_reflectance
    if not 0 <= room_side_reflectance <= 1:
        errors.append(
            f"blinds.room_side_reflectance must be 0 <= r <= 1, got {room_side_reflectance!r}"
        )

    # --- sky (Reinhart subdivisions) ---
    if project.sky.MF not in _MF_VALUES:
        errors.append(f"sky.MF must be one of {sorted(_MF_VALUES)}, got {project.sky.MF!r}")

    # --- analysis.hours (what selects every reported hour) ---
    if not _hours_is_valid(project.analysis.hours):
        errors.append(
            "analysis.hours must look like 'HH:MM-HH:MM' in whole hours, with "
            f"0 <= start < end <= 24, got {project.analysis.hours!r}"
        )
    elif (
        project.analysis.daylight_saving != "none"
        and int(project.analysis.hours[:2]) < _EARLIEST_DST_START_H
    ):
        errors.append(
            f"analysis.hours {project.analysis.hours!r} opens before "
            f"{_EARLIEST_DST_START_H:02d}:00 under daylight_saving "
            f"{project.analysis.daylight_saving!r}; the clocks change at about "
            "02:00, inside the window on the changeover days -- open it at "
            f"{_EARLIEST_DST_START_H:02d}:00 or later, or set daylight_saving to none"
        )

    # --- option2 ---
    if project.option2.lux_min > project.option2.lux_max:
        errors.append(
            "option2.lux_min must be <= option2.lux_max, got lux_min="
            f"{project.option2.lux_min!r}, lux_max={project.option2.lux_max!r}"
        )
    unverified = [
        f"option2.{name}" for name in ("blinds", "time_basis")
        if getattr(project.option2, name) == UNVERIFIED
    ]
    if unverified:
        warnings.append(
            f"{' and '.join(unverified)} {'is' if len(unverified) == 1 else 'are'} "
            f"{UNVERIFIED}; Option 2 cannot be run until it is settled"
        )

    # --- site (I6: a physically impossible value, even a finite one, is a
    # refusal, not just a warning) ---
    if not -90.0 <= project.site.latitude_N <= 90.0:
        errors.append(
            f"site.latitude_N must be -90 <= lat <= 90, got {project.site.latitude_N!r}"
        )
    if not -180.0 <= project.site.longitude_E <= 180.0:
        errors.append(
            f"site.longitude_E must be -180 <= lon <= 180, got {project.site.longitude_E!r}"
        )
    if not -180.0 <= project.site.meridian_radiance <= 180.0:
        errors.append(
            "site.meridian_radiance must be -180 <= m <= 180, got "
            f"{project.site.meridian_radiance!r}"
        )
    if not -360.0 <= project.site.north_rotation_deg <= 360.0:
        errors.append(
            "site.north_rotation_deg must be -360 <= r <= 360, got "
            f"{project.site.north_rotation_deg!r}"
        )

    _check_meridian(project.site.longitude_E, project.site.meridian_radiance, errors, warnings)

    latitude_N = project.site.latitude_N
    if not _LATITUDE_MIN_N <= latitude_N <= _LATITUDE_MAX_N:
        warnings.append(
            f"site.latitude_N is {latitude_N!r}, outside the "
            f"{_LATITUDE_MIN_N} to {_LATITUDE_MAX_N} degrees N band this tool has "
            "been checked against"
        )

    # --- performance ---
    if project.performance.trace_scope == "storey":
        warnings.append(
            "performance.trace_scope is 'storey': this must be shown to equal "
            "'all' (within 0.01 percentage points) on the acceptance fixture "
            "before it is relied on"
        )

    # performance.linked_zones naming a storey with no rooms is deferred to
    # T3.4, which is where rooms exist -- not checked here.

    # --- views (Quality Views spec 5.1) ---
    views = getattr(project, "views", None)
    if views is not None:
        if not 0.5 <= views.eye_height_m <= 2.0:
            errors.append(
                f"views.eye_height_m must be 0.5 <= h <= 2.0 (a seated or standing eye "
                f"height in metres), got {views.eye_height_m!r}"
            )
        if not views.eye_height_source.strip():
            errors.append(
                "views.eye_height_source must say where the eye height came from; "
                f"got {views.eye_height_source!r}"
            )
        if not 0 <= views.max_interior_glass <= 5:
            errors.append(
                f"views.max_interior_glass must be 0 <= n <= 5, got {views.max_interior_glass!r}"
            )
        if not 0.5 <= views.angular_step_deg <= 5.0:
            errors.append(
                f"views.angular_step_deg must be 0.5 <= a <= 5.0, got {views.angular_step_deg!r}"
            )

    # --- atlas drawings (atlas spec 8.1) ---
    atlas = getattr(project, "atlas", None)
    _check_drawings(atlas.drawings if atlas is not None else (), errors)

    return tuple(errors), tuple(warnings)


def _check_drawings(drawings, errors: list[str]) -> None:
    """What the numbers of `atlas.drawings` alone show: opacity, one drawing
    per storey, and two reference points far enough apart. The storey, the
    image path and the image itself are checked where the case is known
    (``deliver.atlas.drawing``)."""
    first_of: dict[str, int] = {}
    for i, drawing in enumerate(drawings):
        key = f"atlas.drawings[{i}]"
        if drawing.storey in first_of:
            errors.append(
                f"{key}: storey {drawing.storey!r} already has a drawing "
                f"(atlas.drawings[{first_of[drawing.storey]}]); give one drawing per storey"
            )
        else:
            first_of[drawing.storey] = i
        if not 0.0 <= drawing.opacity <= 1.0:
            errors.append(f"{key}.opacity must be 0 <= o <= 1, got {drawing.opacity!r}")
        (u0, v0), (u1, v1) = drawing.pixel
        pixels = math.hypot(u1 - u0, v1 - v0)
        if pixels < _MIN_DRAWING_GAP_PX:
            errors.append(
                f"{key}.points: the two pixels are {pixels:g} pixels apart; place them at least "
                f"{_MIN_DRAWING_GAP_PX:g} pixels apart (the farther apart, the more exactly the drawing sits)"
            )
        (x0, y0), (x1, y1) = drawing.model
        metres = math.hypot(x1 - x0, y1 - y0)
        if metres < _MIN_DRAWING_GAP_M:
            errors.append(
                f"{key}.points: the two model points are {metres:g} m apart; place them at least "
                f"{_MIN_DRAWING_GAP_M:g} m apart (the farther apart, the more exactly the drawing sits)"
            )


# --- Option 2 readiness (Option 2 design, sections 3 and 7) ----------------

#: The one time basis Option 2 runs on: 9 a.m. and 3 p.m. as the weather
#: file stamps them, in its own time zone, with no daylight-saving shift.
OPTION2_TIME_BASIS = "weather_file_local"
#: The one sky Option 2 runs on: gendaylit -W from the weather file's own
#: irradiance at that hour.
OPTION2_SKY = "weather_file"
#: The widest search either side of an equinox this tool accepts, in days.
OPTION2_MAX_WINDOW_DAYS = 30

_MMDD_RE = re.compile(r"^(\d{2})-(\d{2})$")
_WHOLE_HOUR_RE = re.compile(r"^(\d{2}):00$")


def parse_mmdd(text: str) -> tuple[int, int]:
    """``"03-21"`` -> ``(3, 21)``. A ``ValueError`` for anything else,
    including a date that does not exist in a non-leap year."""
    match = _MMDD_RE.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"expected a quoted 'MM-DD' date, got {text!r}")
    month, day = int(match.group(1)), int(match.group(2))
    try:
        datetime.date(_WEA_YEAR, month, day)
    except ValueError:
        raise ValueError(f"{text!r} is not a date in a non-leap year") from None
    return month, day


def parse_whole_hour(text: str) -> int:
    """``"09:00"`` -> ``9``. Only a whole hour from 01:00 to 23:00: the
    irradiance at a time is the mean of the two half-hour weather records
    either side of it, so both must exist."""
    match = _WHOLE_HOUR_RE.match(text) if isinstance(text, str) else None
    if match is None or not 1 <= int(match.group(1)) <= 23:
        raise ValueError(
            f"expected a quoted whole hour 'HH:00' from 01:00 to 23:00, got {text!r}"
        )
    return int(match.group(1))


def check_option2(project: "Project") -> tuple[str, ...]:
    """Every reason the ``option2`` command cannot run this project; empty
    when it can. Never raises. ``run`` and ``views`` never call this, so
    they keep loading a project whose Option 2 section is still UNVERIFIED."""
    spec = project.option2
    errors: list[str] = []
    if spec.time_basis != OPTION2_TIME_BASIS:
        errors.append(
            f"option2.time_basis must be {OPTION2_TIME_BASIS!r} (9 a.m. and 3 p.m. as "
            "the weather file stamps them, no daylight-saving shift), got "
            f"{spec.time_basis!r}"
        )
    if spec.sky != OPTION2_SKY:
        errors.append(
            f"option2.sky must be {OPTION2_SKY!r} (Radiance gendaylit from the "
            f"weather file's own irradiance), got {spec.sky!r}"
        )
    if spec.blinds != "exclude":
        errors.append(
            "option2.blinds must be 'exclude' (LEED: exclude blinds or shades from "
            f"the model), got {spec.blinds!r}"
        )
    window = spec.clearest_day_window_days
    window_ok = 0 <= window <= OPTION2_MAX_WINDOW_DAYS
    if not window_ok:
        errors.append(
            "option2.clearest_day_window_days must be from 0 to "
            f"{OPTION2_MAX_WINDOW_DAYS}, got {window!r}"
        )
    if len(spec.equinox_dates) != 2:
        errors.append(
            "option2.equinox_dates must hold exactly two dates, got "
            f"{len(spec.equinox_dates)}: {list(spec.equinox_dates)!r}"
        )
    months = []
    for text in spec.equinox_dates:
        try:
            month, day = parse_mmdd(text)
        except ValueError as exc:
            errors.append(f"option2.equinox_dates: {exc}")
            continue
        months.append(month)
        if not window_ok:
            continue
        centre = datetime.date(_WEA_YEAR, month, day)
        span = datetime.timedelta(days=window)
        if (centre - span).year != _WEA_YEAR:
            errors.append(
                f"option2.equinox_dates: {text} with a {window}-day window runs past "
                "1 January; the weather file holds one calendar year"
            )
        if (centre + span).year != _WEA_YEAR:
            errors.append(
                f"option2.equinox_dates: {text} with a {window}-day window runs past "
                "31 December; the weather file holds one calendar year"
            )
    if len(months) == 2 and months != [3, 9]:
        errors.append(
            "option2.equinox_dates must be one date in March, then one in September "
            f"(LEED: one day near each equinox), got {list(spec.equinox_dates)!r}"
        )
    if not spec.times:
        errors.append("option2.times must hold at least one time")
    for text in spec.times:
        try:
            parse_whole_hour(text)
        except ValueError as exc:
            errors.append(f"option2.times: {exc}")
    if len(set(spec.times)) != len(spec.times):
        errors.append(f"option2.times must not repeat a time, got {list(spec.times)!r}")
    return tuple(errors)
