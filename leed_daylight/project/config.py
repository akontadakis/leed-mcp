# leed_daylight/project/config.py
"""Load project.yaml into one frozen ``Project``.

Loading, typing and vocabulary only: is a key present, is it the right
type, is a string one of the permitted words. Cross-field checks, value
ranges and filesystem existence are the next layer's job (T3.2), so this
module never opens a file it is only pointed at, and it carries no numeric
default anywhere -- a missing input is always a refusal, never a guess.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime
import math
import pathlib
import re
import shlex
import types
from collections.abc import Mapping, Sequence

import yaml

from leed_daylight.constants import FILL_ME
from leed_daylight.units import LENGTH_TO_M

from ..scene.input_file import read_input_text
from .checks import _check_file, check_project

#: The 14 top-level sections project.yaml must have (R-P3/R-P6).
_TOP_LEVEL_SECTIONS = (
    "project", "site", "scene", "standard", "weather", "glazing",
    "reflectance", "grid", "analysis", "performance", "sky", "radiance",
    "blinds", "option2",
)

#: Optional top-level sections: absent means "this credit is not run".
_OPTIONAL_SECTIONS = ("views", "report", "atlas", "calculator")

#: Atlas spec 8.1: a drawing under a storey plan is this opaque unless its
#: entry says otherwise.
ATLAS_DEFAULT_OPACITY = 0.35

#: Decision 39 (user, 23 Sep 2026): the interior-glass limit is a setting
#: with default 1, the ClimateStudio default; the panorama step default is
#: the ray spacing the spec was sized on. Neither comes from client data.
#: `views.eye_height_m` has no default: it is always stated.
VIEWS_DEFAULT_MAX_INTERIOR_GLASS = 1
VIEWS_DEFAULT_ANGULAR_STEP_DEG = 2.0

#: Figure renderers the report step can use (reports spec R3); the first is
#: the default.
RENDERERS = ("matplotlib", "cairosvg", "pillow")

_UNITS = frozenset(LENGTH_TO_M)
_STANDARDS = frozenset({"LM-83-12", "LM-83-23"})
_FRAME_STYLES = frozenset({"constant_width", "uniform_scale"})
_REFLECTANCE_POLICIES = frozenset({"override", "fill_gaps", "keep_as_modelled"})
_DAYLIGHT_SAVINGS = frozenset({"EU", "US", "none"})
_AGGREGATIONS = frozenset({"floor_area", "points"})
_MOVEABLE_FURNITURE = frozenset({"include", "exclude"})
_TRACE_SCOPES = frozenset({"storey", "all"})
_TRIGGER_AREAS = frozenset({"room", "grid"})

#: The three glass classes of LM-83-12 clause 2.2.9, by tilt from flat
#: (skylights spec D2): the keys a `glazing.dirt_factor` mapping may use.
DIRT_CLASSES: tuple[str, ...] = ("vertical", "sloped", "skylight")
#: `glazing.frame_share` keys that name a glass class, not a role (D3).
_FRAME_SHARE_CLASS_KEYS = frozenset({"skylight", "sloped"})
#: `glazing.skylight_well: modelled`: the scene carries every well itself.
_WELL_MODELLED = "modelled"


class ProjectError(ValueError):
    """Every refusal raised while loading project.yaml (layer 2)."""


@dataclasses.dataclass(frozen=True)
class Site:
    latitude_N: float
    longitude_E: float
    meridian_radiance: float
    north_rotation_deg: float
    north_rotation_source: str


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    units: str
    files: tuple[pathlib.Path, ...]


@dataclasses.dataclass(frozen=True)
class Analysis:
    hours: str
    daylight_saving: str
    aggregation: str
    moveable_furniture: str


@dataclasses.dataclass(frozen=True)
class Glazing:
    #: One number for every exterior pane, or a mapping from keys of
    #: ``DIRT_CLASSES`` to a number (skylights spec 4.2).
    dirt_factor: float | types.MappingProxyType
    frame_share: types.MappingProxyType
    frame_reflectance: float
    frame_style: str
    reveal_depth_m: float
    reveal_reflectance: float
    vt_includes_dirt: bool
    #: ``"modelled"`` (the scene carries its wells), a well depth in metres,
    #: or None when not given (needed only with skylight-class glass).
    skylight_well: str | float | None = None
    #: The built well's reflectance; only with a well depth.
    skylight_well_reflectance: float | None = None

    def dirt_for(self, tilt_class: str) -> float:
        """The dirt factor for one glass class. The single-number form
        serves every class; the mapping form raises ``KeyError(tilt_class)``
        for a class it does not hold."""
        if isinstance(self.dirt_factor, Mapping):
            return self.dirt_factor[tilt_class]
        return self.dirt_factor


@dataclasses.dataclass(frozen=True)
class Reflectance:
    policy: str
    by_role: types.MappingProxyType


@dataclasses.dataclass(frozen=True)
class Grid:
    spacing_m: float
    work_plane_m: float
    min_clearance_m: float
    #: ``grid.points_file``, resolved against the project folder: the
    #: sensor points to use instead of the lattice (``project/points.py``).
    #: None (the key left out) means the lattice.
    points_file: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class Blinds:
    transmittance_diffuse: float
    transmittance_specular: float
    room_side_reflectance: float
    #: ``blinds.trigger_area``: the area that picks a space's LM-83-12
    #: 2.2.6 tier, ``room`` (the room area from the scene) or ``grid`` (the
    #: sensor point count times ``grid.spacing_m`` squared).
    trigger_area: str = "room"


@dataclasses.dataclass(frozen=True)
class Sky:
    basis: str
    MF: int


@dataclasses.dataclass(frozen=True)
class RadianceParams:
    ambient_draft: str
    ambient_full: str
    direct: str


@dataclasses.dataclass(frozen=True)
class Performance:
    trace_scope: str
    linked_zones: types.MappingProxyType


@dataclasses.dataclass(frozen=True)
class Option2Spec:
    equinox_dates: tuple[str, ...]
    clearest_day_window_days: int
    times: tuple[str, ...]
    sky: str
    lux_min: float
    lux_max: float
    blinds: str
    time_basis: str


@dataclasses.dataclass(frozen=True)
class ViewsSpec:
    eye_height_m: float
    eye_height_source: str
    max_interior_glass: int
    angular_step_deg: float


@dataclasses.dataclass(frozen=True)
class ReportSpec:
    """The optional `report:` section (reports spec 3.3)."""
    renderer: str = RENDERERS[0]
    deviations: tuple[str, ...] = ()
    prepared_by: str | None = None


@dataclasses.dataclass(frozen=True)
class DrawingSpec:
    """One drawing under a storey plan (atlas spec 8.1). ``image`` is a
    path relative to the project folder; ``pixel`` holds two points of the
    image (pixels, x to the right, y down) and ``model`` the two model
    points they show (metres, as the prepared model, y up)."""
    storey: str
    image: str
    pixel: tuple[tuple[float, float], tuple[float, float]]
    model: tuple[tuple[float, float], tuple[float, float]]
    opacity: float = ATLAS_DEFAULT_OPACITY


@dataclasses.dataclass(frozen=True)
class AtlasSpec:
    """The optional `atlas:` section (atlas spec 8.1)."""
    drawings: tuple[DrawingSpec, ...] = ()


#: The calculator's two unit systems (calculator spec D2): "IP units" or "SI units" in its Instructions sheet.
CALCULATOR_UNITS = ("IP", "SI")


@dataclasses.dataclass(frozen=True)
class CalculatorSpec:
    """The optional `calculator:` section (calculator spec D2). `units` has no
    default: it must match the project's LEED registration."""
    units: str | None = None


@dataclasses.dataclass(frozen=True)
class Project:
    name: str
    credit: str
    root: pathlib.Path
    site: Site
    scene: SceneSpec
    standard: str
    credit_rules_path: pathlib.Path
    weather_file: pathlib.Path
    glazing: Glazing
    reflectance: Reflectance
    grid: Grid
    analysis: Analysis
    blinds: Blinds
    sky: Sky
    radiance: RadianceParams
    performance: Performance
    option2: Option2Spec
    warnings: tuple[str, ...]
    #: Quality Views settings; None when project.yaml has no `views:` section.
    views: ViewsSpec | None = None
    #: Report settings; the defaults when project.yaml has no `report:` section.
    report: ReportSpec = ReportSpec()
    #: Atlas settings; no drawing when project.yaml has no `atlas:` section.
    atlas: AtlasSpec = AtlasSpec()
    #: Calculator settings; units None when project.yaml has no `calculator:` section.
    calculator: CalculatorSpec = CalculatorSpec()


def _fail(detail: str) -> None:
    raise ProjectError(f"project.yaml: {detail}") from None


def _missing(path: str) -> None:
    raise ProjectError(f"project.yaml: missing required key '{path}'") from None


def _get(section: dict, key: str, path: str):
    """Fetch a required key from an already-typed mapping, or refuse."""
    if not isinstance(section, dict) or key not in section:
        _missing(path)
    return section[key]


def _get_section(container: dict, key: str, path: str) -> dict:
    """Fetch a required key that must itself be a mapping.

    Every section and nested sub-section in project.yaml is a mapping
    (a YAML "block", never a list or a scalar). This is the one place that
    checks that shape, so a malformed section (a list, a string, a number)
    is always refused by name here, before any code that assumes it can
    call ``.items()`` or iterate its keys ever sees it.
    """
    value = _get(container, key, path)
    if not isinstance(value, dict):
        _fail(f"{path} must be a mapping, got {value!r}")
    return value


def _check_closed(section: dict, known: frozenset, section_name: str) -> None:
    """Refuse an unknown key inside a CLOSED section (R-P4).

    ``section`` must already be known to be a mapping -- every caller
    fetches it through ``_get_section`` first.
    """
    extra = set(section) - set(known)
    if extra:
        bad = sorted(extra)[0]
        _fail(f"unknown key '{section_name}.{bad}'")


def _check_role_keys(section: dict, allowed: frozenset, special: frozenset, path: str) -> None:
    """Refuse a key in a role-keyed mapping that is not a role (or special key).

    ``reflectance.*`` and ``glazing.frame_share.*`` are keyed by role names
    from materials.csv. A misspelt role (``celing``, ``curtain_wall``) would
    otherwise match no modifier and be ignored without a word, so every key
    must be one of ``allowed`` or ``special``.
    """
    valid = sorted(special) + sorted(allowed)
    for key in section:
        if key not in allowed and key not in special:
            _fail(
                f"unknown key '{path}.{key}': not a role name; "
                f"valid names are {', '.join(valid)}"
            )


#: Keys whose values are clock times or dates. Unquoted, YAML 1.1 reads a
#: time such as 15:00 as the base-60 integer 900.
_TIME_LIKE_KEYS = frozenset({"analysis.hours", "option2.times[]", "option2.equinox_dates[]"})


def _string_hint(value, path: str) -> str:
    """Explain the YAML 1.1 reading behind a non-string, or return ''."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return (
            " (YAML reads an unquoted date as a date, not text; quote it, "
            f"e.g. \"{value.isoformat()}\")"
        )
    if path in _TIME_LIKE_KEYS and isinstance(value, int) and not isinstance(value, bool):
        hours, minutes = divmod(value, 60)
        return (
            f" (YAML reads an unquoted clock time such as {hours}:{minutes:02d} as the "
            f"base-60 number {value}; quote it: \"{hours}:{minutes:02d}\")"
        )
    return ""


def _number_hint(value) -> str:
    """Explain the YAML 1.1 reading of a numeric-looking string, or return ''."""
    if not isinstance(value, str):
        return ""
    try:
        float(value)
    except ValueError:
        return ""
    return " (YAML reads 1e3 as text; write 1.0e3)"


def _as_str(value, path: str) -> str:
    if not isinstance(value, str):
        _fail(f"{path} must be a string, got {value!r}{_string_hint(value, path)}")
    return value


def _as_float(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be a number, got {value!r}{_number_hint(value)}")
    v = float(value)
    if not math.isfinite(v):
        # I6: YAML spells these `.nan`/`.inf`; left unchecked, one reaches
        # the manifest as bare, invalid JSON (`north_rotation_deg`) or
        # silently disables a feature that tests `> 0.0` (`reveal_depth_m`).
        _fail(f"{path} must be a finite number, got {value!r}")
    return v


def _as_int(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path} must be an integer, got {value!r}{_number_hint(value)}")
    return value


def _as_bool(value, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path} must be a boolean, got {value!r}")
    return value


def _as_enum(value, permitted: frozenset, path: str) -> str:
    value = _as_str(value, path)
    if value not in permitted:
        _fail(f"{path} is {value!r}; expected one of {sorted(permitted)}")
    return value


def _as_str_tuple(value, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(f"{path} must be a list, got {value!r}")
    return tuple(_as_str(item, f"{path}[]") for item in value)


def _resolve(root: pathlib.Path, value, path: str) -> pathlib.Path:
    return root / _as_str(value, path)


def _load_site(raw: dict) -> Site:
    section = _get_section(raw, "site", "site")
    _check_closed(
        section,
        frozenset({
            "latitude_N", "longitude_E", "meridian_radiance",
            "north_rotation_deg", "north_rotation_source",
        }),
        "site",
    )
    return Site(
        latitude_N=_as_float(_get(section, "latitude_N", "site.latitude_N"), "site.latitude_N"),
        longitude_E=_as_float(_get(section, "longitude_E", "site.longitude_E"), "site.longitude_E"),
        meridian_radiance=_as_float(
            _get(section, "meridian_radiance", "site.meridian_radiance"), "site.meridian_radiance"
        ),
        north_rotation_deg=_as_float(
            _get(section, "north_rotation_deg", "site.north_rotation_deg"), "site.north_rotation_deg"
        ),
        north_rotation_source=_as_str(
            _get(section, "north_rotation_source", "site.north_rotation_source"),
            "site.north_rotation_source",
        ),
    )


def _load_scene(raw: dict, root: pathlib.Path) -> SceneSpec:
    section = _get_section(raw, "scene", "scene")
    _check_closed(section, frozenset({"units", "files"}), "scene")
    units = _as_enum(_get(section, "units", "scene.units"), _UNITS, "scene.units")
    files = _get(section, "files", "scene.files")
    if not isinstance(files, list):
        _fail(f"scene.files must be a list, got {files!r}")
    paths = tuple(_resolve(root, f, f"scene.files[{i}]") for i, f in enumerate(files))
    return SceneSpec(units=units, files=paths)


def _load_glazing(raw: dict) -> Glazing:
    section = _get_section(raw, "glazing", "glazing")
    _check_closed(
        section,
        frozenset({
            "dirt_factor", "frame_share", "frame_reflectance", "frame_style",
            "reveal_depth_m", "reveal_reflectance", "vt_includes_dirt",
            "skylight_well", "skylight_well_reflectance",
        }),
        "glazing",
    )
    # Imported here: materials.py imports ProjectError from this module.
    from .materials import EXTERIOR_GLAZING_ROLES

    frame_share_raw = _get_section(section, "frame_share", "glazing.frame_share")
    _check_role_keys(
        frame_share_raw,
        EXTERIOR_GLAZING_ROLES,
        frozenset({"default"}) | _FRAME_SHARE_CLASS_KEYS,
        "glazing.frame_share",
    )
    if "default" not in frame_share_raw:
        _missing("glazing.frame_share.default")
    frame_share = {}
    for key, value in frame_share_raw.items():
        # Keys are exterior-glazing role names (checked above), and a
        # value of null is typed as-is -- T3.2 refuses it, not this module.
        frame_share[key] = None if value is None else _as_float(value, f"glazing.frame_share.{key}")

    return Glazing(
        dirt_factor=_load_dirt_factor(_get(section, "dirt_factor", "glazing.dirt_factor")),
        frame_share=types.MappingProxyType(frame_share),
        frame_reflectance=_as_float(
            _get(section, "frame_reflectance", "glazing.frame_reflectance"), "glazing.frame_reflectance"
        ),
        frame_style=_as_enum(
            _get(section, "frame_style", "glazing.frame_style"), _FRAME_STYLES, "glazing.frame_style"
        ),
        reveal_depth_m=_as_float(
            _get(section, "reveal_depth_m", "glazing.reveal_depth_m"), "glazing.reveal_depth_m"
        ),
        reveal_reflectance=_as_float(
            _get(section, "reveal_reflectance", "glazing.reveal_reflectance"), "glazing.reveal_reflectance"
        ),
        vt_includes_dirt=_as_bool(
            _get(section, "vt_includes_dirt", "glazing.vt_includes_dirt"), "glazing.vt_includes_dirt"
        ),
        skylight_well=(
            _load_skylight_well(section["skylight_well"]) if "skylight_well" in section else None
        ),
        skylight_well_reflectance=(
            _as_float(section["skylight_well_reflectance"], "glazing.skylight_well_reflectance")
            if "skylight_well_reflectance" in section
            else None
        ),
    )


def _load_dirt_factor(value) -> float | types.MappingProxyType:
    """One number, or a mapping from glass class to a number (D2). The
    ranges are checked in ``checks.py``; which classes the scene needs is
    checked in ``build.prepare``, the first place that sees the scene."""
    if not isinstance(value, dict):
        return _as_float(value, "glazing.dirt_factor")
    if not value:
        _fail(
            "glazing.dirt_factor is an empty mapping; give one number, or one "
            f"value per glass class from {', '.join(DIRT_CLASSES)}"
        )
    for key in value:
        if key not in DIRT_CLASSES:
            _fail(
                f"unknown key 'glazing.dirt_factor.{key}'; valid keys are "
                f"{', '.join(DIRT_CLASSES)}"
            )
    return types.MappingProxyType(
        {key: _as_float(item, f"glazing.dirt_factor.{key}") for key, item in value.items()}
    )


def _load_skylight_well(value) -> str | float:
    """``"modelled"`` or a depth in metres (D4). The depth's range is
    checked in ``checks.py``."""
    if isinstance(value, str):
        if value != _WELL_MODELLED:
            _fail(
                f"glazing.skylight_well is {value!r}; expected {_WELL_MODELLED!r} "
                "or a well depth in metres"
            )
        return value
    if value is None:
        _fail(
            f"glazing.skylight_well must be {_WELL_MODELLED!r} or a well depth in "
            "metres, got None"
        )
    return _as_float(value, "glazing.skylight_well")


def _load_reflectance(raw: dict) -> Reflectance:
    from .materials import ROLES

    section = _get_section(raw, "reflectance", "reflectance")
    _check_role_keys(section, ROLES, frozenset({"policy"}), "reflectance")
    policy = _as_enum(
        _get(section, "policy", "reflectance.policy"), _REFLECTANCE_POLICIES, "reflectance.policy"
    )
    # Every other key is a role name from materials.py ROLES (checked above).
    by_role = {
        key: _as_float(value, f"reflectance.{key}")
        for key, value in section.items()
        if key != "policy"
    }
    return Reflectance(policy=policy, by_role=types.MappingProxyType(by_role))


def _load_grid(raw: dict, root: pathlib.Path) -> Grid:
    section = _get_section(raw, "grid", "grid")
    _check_closed(
        section, frozenset({"spacing_m", "work_plane_m", "min_clearance_m", "points_file"}), "grid"
    )
    return Grid(
        spacing_m=_as_float(_get(section, "spacing_m", "grid.spacing_m"), "grid.spacing_m"),
        work_plane_m=_as_float(_get(section, "work_plane_m", "grid.work_plane_m"), "grid.work_plane_m"),
        min_clearance_m=_as_float(
            _get(section, "min_clearance_m", "grid.min_clearance_m"), "grid.min_clearance_m"
        ),
        points_file=(
            _resolve(root, section["points_file"], "grid.points_file")
            if "points_file" in section
            else None
        ),
    )


def _load_analysis(raw: dict) -> Analysis:
    section = _get_section(raw, "analysis", "analysis")
    _check_closed(
        section,
        frozenset({"hours", "daylight_saving", "aggregation", "moveable_furniture"}),
        "analysis",
    )
    return Analysis(
        hours=_as_str(_get(section, "hours", "analysis.hours"), "analysis.hours"),
        daylight_saving=_as_enum(
            _get(section, "daylight_saving", "analysis.daylight_saving"),
            _DAYLIGHT_SAVINGS,
            "analysis.daylight_saving",
        ),
        aggregation=_as_enum(
            _get(section, "aggregation", "analysis.aggregation"), _AGGREGATIONS, "analysis.aggregation"
        ),
        moveable_furniture=_as_enum(
            _get(section, "moveable_furniture", "analysis.moveable_furniture"),
            _MOVEABLE_FURNITURE,
            "analysis.moveable_furniture",
        ),
    )


def _load_blinds(raw: dict) -> Blinds:
    section = _get_section(raw, "blinds", "blinds")
    _check_closed(
        section,
        frozenset({
            "transmittance_diffuse", "transmittance_specular", "room_side_reflectance", "trigger_area",
        }),
        "blinds",
    )
    return Blinds(
        transmittance_diffuse=_as_float(
            _get(section, "transmittance_diffuse", "blinds.transmittance_diffuse"),
            "blinds.transmittance_diffuse",
        ),
        transmittance_specular=_as_float(
            _get(section, "transmittance_specular", "blinds.transmittance_specular"),
            "blinds.transmittance_specular",
        ),
        room_side_reflectance=_as_float(
            _get(section, "room_side_reflectance", "blinds.room_side_reflectance"),
            "blinds.room_side_reflectance",
        ),
        trigger_area=(
            _as_enum(section["trigger_area"], _TRIGGER_AREAS, "blinds.trigger_area")
            if "trigger_area" in section
            else "room"
        ),
    )


def _load_sky(raw: dict) -> Sky:
    section = _get_section(raw, "sky", "sky")
    _check_closed(section, frozenset({"basis", "MF"}), "sky")
    return Sky(
        basis=_as_str(_get(section, "basis", "sky.basis"), "sky.basis"),
        MF=_as_int(_get(section, "MF", "sky.MF"), "sky.MF"),
    )


# The radiance strings are split into rcontrib and rtrace arguments, so only
# numeric options pass, each with its count of numbers. Anything else could
# name a file or a command: -o '!cmd' runs a shell, -af and -e write a file,
# @file reads one.
_RADIANCE_OPTIONS = {
    **{opt: 1 for opt in ("-ab", "-ad", "-as", "-ar", "-aa", "-lw", "-lr",
                          "-dj", "-ds", "-dt", "-dc", "-dr", "-dp", "-st", "-ss")},
    "-av": 3,
}
_RADIANCE_SWITCHES = frozenset({"-u", "-u+", "-u-"})


def _as_radiance_options(value, path: str, *, only: tuple[str, ...] = ()) -> str:
    """``only``, when given, limits the options to those starting with one
    of its prefixes (the switches stay allowed)."""
    text = _as_str(value, path)
    try:
        words = shlex.split(text)
    except ValueError as exc:
        _fail(f"{path} cannot be read as options ({exc})")
    i = 0
    while i < len(words):
        word = words[i]
        if word in _RADIANCE_SWITCHES:
            i += 1
            continue
        count = _RADIANCE_OPTIONS.get(word)
        if count is not None and only and not word.startswith(only):
            _fail(f"{path}: option {word!r} is not allowed here; only the "
                  f"{' and '.join(only)} options are (the ambient options go in "
                  "radiance.ambient)")
        if count is None:
            _fail(f"{path}: option {word!r} is not allowed; use the numeric options "
                  f"{', '.join(sorted(_RADIANCE_OPTIONS))} and -u")
        numbers = words[i + 1:i + 1 + count]
        if len(numbers) < count:
            _fail(f"{path}: {word} needs {count} number(s) after it")
        for number in numbers:
            try:
                ok = math.isfinite(float(number))
            except ValueError:
                ok = False
            if not ok:
                _fail(f"{path}: {word} needs a number, got {number!r}")
        i += 1 + count
    return text


def _load_radiance(raw: dict) -> RadianceParams:
    section = _get_section(raw, "radiance", "radiance")
    _check_closed(section, frozenset({"ambient", "direct"}), "radiance")
    ambient = _get_section(section, "ambient", "radiance.ambient")
    _check_closed(ambient, frozenset({"draft", "full"}), "radiance.ambient")
    return RadianceParams(
        ambient_draft=_as_radiance_options(_get(ambient, "draft", "radiance.ambient.draft"),
                                           "radiance.ambient.draft"),
        ambient_full=_as_radiance_options(_get(ambient, "full", "radiance.ambient.full"),
                                          "radiance.ambient.full"),
        direct=_as_radiance_options(_get(section, "direct", "radiance.direct"), "radiance.direct",
                                    only=("-d", "-s")),
    )


def _load_performance(raw: dict) -> Performance:
    section = _get_section(raw, "performance", "performance")
    _check_closed(section, frozenset({"trace_scope", "linked_zones"}), "performance")
    trace_scope = _as_enum(
        _get(section, "trace_scope", "performance.trace_scope"), _TRACE_SCOPES, "performance.trace_scope"
    )
    linked_zones_raw = _get_section(section, "linked_zones", "performance.linked_zones")
    # OPEN mapping (R-P4): keys are storey names.
    linked_zones = {
        key: _as_str_tuple(value, f"performance.linked_zones.{key}")
        for key, value in linked_zones_raw.items()
    }
    return Performance(trace_scope=trace_scope, linked_zones=types.MappingProxyType(linked_zones))


def _load_option2(raw: dict) -> Option2Spec:
    section = _get_section(raw, "option2", "option2")
    _check_closed(
        section,
        frozenset({
            "equinox_dates", "clearest_day_window_days", "times", "sky",
            "lux_min", "lux_max", "blinds", "time_basis",
        }),
        "option2",
    )
    return Option2Spec(
        equinox_dates=_as_str_tuple(
            _get(section, "equinox_dates", "option2.equinox_dates"), "option2.equinox_dates"
        ),
        clearest_day_window_days=_as_int(
            _get(section, "clearest_day_window_days", "option2.clearest_day_window_days"),
            "option2.clearest_day_window_days",
        ),
        times=_as_str_tuple(_get(section, "times", "option2.times"), "option2.times"),
        sky=_as_str(_get(section, "sky", "option2.sky"), "option2.sky"),
        lux_min=_as_float(_get(section, "lux_min", "option2.lux_min"), "option2.lux_min"),
        lux_max=_as_float(_get(section, "lux_max", "option2.lux_max"), "option2.lux_max"),
        blinds=_as_str(_get(section, "blinds", "option2.blinds"), "option2.blinds"),
        time_basis=_as_str(_get(section, "time_basis", "option2.time_basis"), "option2.time_basis"),
    )


def _load_views(raw: dict) -> ViewsSpec | None:
    if "views" not in raw:
        return None
    section = _get_section(raw, "views", "views")
    _check_closed(
        section,
        frozenset({"eye_height_m", "eye_height_source", "max_interior_glass", "angular_step_deg"}),
        "views",
    )
    return ViewsSpec(
        eye_height_m=_as_float(_get(section, "eye_height_m", "views.eye_height_m"), "views.eye_height_m"),
        eye_height_source=_as_str(
            _get(section, "eye_height_source", "views.eye_height_source"), "views.eye_height_source"
        ),
        max_interior_glass=_as_int(
            section.get("max_interior_glass", VIEWS_DEFAULT_MAX_INTERIOR_GLASS),
            "views.max_interior_glass",
        ),
        angular_step_deg=_as_float(
            section.get("angular_step_deg", VIEWS_DEFAULT_ANGULAR_STEP_DEG),
            "views.angular_step_deg",
        ),
    )


def _load_report(raw: dict) -> ReportSpec:
    if "report" not in raw:
        return ReportSpec()
    section = _get_section(raw, "report", "report")
    _check_closed(section, frozenset({"renderer", "deviations", "prepared_by"}), "report")
    return ReportSpec(
        renderer=_as_enum(section.get("renderer", RENDERERS[0]), frozenset(RENDERERS), "report.renderer"),
        deviations=_as_str_tuple(section.get("deviations", []), "report.deviations"),
        prepared_by=(
            _as_str(section["prepared_by"], "report.prepared_by") if "prepared_by" in section else None
        ),
    )


def _as_pair(value, path: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        _fail(f"{path} must be a list of two numbers, got {value!r}")
    return (_as_float(value[0], f"{path}[0]"), _as_float(value[1], f"{path}[1]"))


def _load_drawing(entry, path: str) -> DrawingSpec:
    """One `atlas.drawings` entry: its shape only. Ranges, the two points'
    spacing and one drawing per storey are checked in ``checks.py``; the
    storey, the image path and the image itself in ``deliver.atlas.drawing``."""
    if not isinstance(entry, dict):
        _fail(f"{path} must be a mapping, got {entry!r}")
    _check_closed(entry, frozenset({"storey", "image", "points", "opacity"}), path)
    points = _get(entry, "points", f"{path}.points")
    if not isinstance(points, list) or len(points) != 2:
        _fail(f"{path}.points must be a list of exactly two points, got {points!r}")
    pixel, model = [], []
    for j, point in enumerate(points):
        where = f"{path}.points[{j}]"
        if not isinstance(point, dict):
            _fail(f"{where} must be a mapping with a pixel and a model point, got {point!r}")
        _check_closed(point, frozenset({"pixel", "model"}), where)
        pixel.append(_as_pair(_get(point, "pixel", f"{where}.pixel"), f"{where}.pixel"))
        model.append(_as_pair(_get(point, "model", f"{where}.model"), f"{where}.model"))
    return DrawingSpec(
        storey=_as_str(_get(entry, "storey", f"{path}.storey"), f"{path}.storey"),
        image=_as_str(_get(entry, "image", f"{path}.image"), f"{path}.image"),
        pixel=(pixel[0], pixel[1]),
        model=(model[0], model[1]),
        opacity=_as_float(entry.get("opacity", ATLAS_DEFAULT_OPACITY), f"{path}.opacity"),
    )


def _load_atlas(raw: dict) -> AtlasSpec:
    if "atlas" not in raw:
        return AtlasSpec()
    section = _get_section(raw, "atlas", "atlas")
    _check_closed(section, frozenset({"drawings"}), "atlas")
    entries = section.get("drawings", [])
    if not isinstance(entries, list):
        _fail(f"atlas.drawings must be a list, got {entries!r}")
    return AtlasSpec(
        drawings=tuple(_load_drawing(entry, f"atlas.drawings[{i}]") for i, entry in enumerate(entries))
    )


def _load_calculator(raw: dict) -> CalculatorSpec:
    if "calculator" not in raw:
        return CalculatorSpec()
    section = _get_section(raw, "calculator", "calculator")
    _check_closed(section, frozenset({"units"}), "calculator")
    if "units" not in section:
        return CalculatorSpec()
    return CalculatorSpec(units=_as_enum(section["units"], frozenset(CALCULATOR_UNITS), "calculator.units"))


def _fill_me_paths(value, path: str) -> list[str]:
    """Every dotted path under ``value`` whose value is the FILL_ME marker
    (spaces around it ignored), in file order; list entries carry their
    index, e.g. ``scene.files[0]``."""
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            found += _fill_me_paths(item, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found += _fill_me_paths(item, f"{path}[{index}]")
        return found
    if isinstance(value, str) and value.strip() == FILL_ME:
        return [path]
    return []


def _refuse_fill_me(raw: dict) -> None:
    """CLI spec 5: refuse, listing every one, any value still marked
    FILL_ME -- a text value would otherwise pass its type check."""
    paths = _fill_me_paths(raw, "")
    if paths:
        _fail(f"{len(paths)} value(s) still say {FILL_ME}; fill them in: {', '.join(paths)}")


def load_project(path: str | pathlib.Path) -> Project:
    """Load and validate project.yaml, returning one frozen ``Project``.

    Either a fully valid ``Project`` comes back, or ``ProjectError`` is
    raised; there is no partly valid state. Relative paths inside the file
    resolve against ``root``, the directory the file sits in -- never
    against the current working directory.
    """
    path = pathlib.Path(path).absolute()
    root = path.parent

    raw = yaml.safe_load(read_input_text(path))
    if not isinstance(raw, dict):
        _fail("project.yaml must be a mapping at the top level")
    _refuse_fill_me(raw)
    project = _typed_project(raw, root)

    # Cross-field and filesystem checks (T3.2). One bad key must not hide
    # the other nine, so every check runs before any of them can refuse.
    check_errors, check_warnings = check_project(project)
    if check_errors:
        _fail(
            f"{len(check_errors)} problem(s) found by check_project:\n  "
            + "\n  ".join(check_errors)
        )
    return dataclasses.replace(project, warnings=project.warnings + check_warnings)


def _typed_project(raw: dict, root: pathlib.Path) -> Project:
    """The typed load of ``load_project``, before ``check_project``: every
    section present, every key typed, in one fixed order, so the first
    refusal is always the same one. ``check_project_draft`` runs it too."""
    warnings = []
    unknown_top_level = sorted(set(raw) - set(_TOP_LEVEL_SECTIONS) - set(_OPTIONAL_SECTIONS))
    for name in unknown_top_level:
        warnings.append(f"unknown top-level section '{name}' (ignored)")

    for section in _TOP_LEVEL_SECTIONS:
        if section not in raw:
            _missing(section)

    project_section = _get_section(raw, "project", "project")
    _check_closed(project_section, frozenset({"name", "credit"}), "project")

    standard_section = _get_section(raw, "standard", "standard")
    _check_closed(standard_section, frozenset({"daylight", "credit_rules"}), "standard")

    weather_section = _get_section(raw, "weather", "weather")
    _check_closed(weather_section, frozenset({"file"}), "weather")

    project = Project(
        name=_as_str(_get(project_section, "name", "project.name"), "project.name"),
        credit=_as_str(_get(project_section, "credit", "project.credit"), "project.credit"),
        root=root,
        site=_load_site(raw),
        scene=_load_scene(raw, root),
        standard=_as_enum(
            _get(standard_section, "daylight", "standard.daylight"), _STANDARDS, "standard.daylight"
        ),
        credit_rules_path=_resolve(
            root, _get(standard_section, "credit_rules", "standard.credit_rules"), "standard.credit_rules"
        ),
        weather_file=_resolve(root, _get(weather_section, "file", "weather.file"), "weather.file"),
        glazing=_load_glazing(raw),
        reflectance=_load_reflectance(raw),
        grid=_load_grid(raw, root),
        analysis=_load_analysis(raw),
        blinds=_load_blinds(raw),
        sky=_load_sky(raw),
        radiance=_load_radiance(raw),
        performance=_load_performance(raw),
        option2=_load_option2(raw),
        views=_load_views(raw),
        report=_load_report(raw),
        atlas=_load_atlas(raw),
        calculator=_load_calculator(raw),
        warnings=tuple(warnings),
    )
    return project


# --- The wizard's draft check and scene-only reader (wizard spec 7.3) -------

#: What stands in for a value still FILL_ME while a draft is checked: a value
#: of the type the typed load wants, chosen so it adds no error of its own to
#: a filled key. Keyed by the value's path with list indices written ``[]``
#: and role, glass-class and storey keys written ``*`` (``_pattern``). A
#: stand-in lives only in the in-memory copy ``check_project_draft`` types:
#: it is never written, returned or logged, and every check error that names
#: the unfinished path is dropped.
_DRAFT_STAND_INS: Mapping[str, object] = types.MappingProxyType({
    "project.name": "draft",
    "project.credit": "draft",
    "site.latitude_N": 0.0,
    "site.longitude_E": 0.0,
    "site.meridian_radiance": 0.0,
    "site.north_rotation_deg": 0.0,
    "site.north_rotation_source": "draft",
    "scene.units": "m",
    "scene.files": [],
    "scene.files[]": "draft.rad",
    "standard.daylight": "LM-83-12",
    "standard.credit_rules": "draft.yaml",
    "weather.file": "draft.wea",
    "glazing.dirt_factor": 1.0,
    "glazing.dirt_factor.*": 1.0,
    "glazing.frame_share": {"default": 0.0},
    "glazing.frame_share.*": 0.0,
    "glazing.frame_reflectance": 0.0,
    "glazing.frame_style": "constant_width",
    "glazing.reveal_depth_m": 0.0,
    "glazing.reveal_reflectance": 0.0,
    "glazing.vt_includes_dirt": False,
    "glazing.skylight_well": _WELL_MODELLED,
    "glazing.skylight_well_reflectance": 0.0,
    "reflectance.policy": "override",
    "reflectance.*": 0.0,
    "grid.spacing_m": 0.5,
    "grid.work_plane_m": 1.0,
    "grid.min_clearance_m": 0.0,
    "grid.points_file": "draft.csv",
    "analysis.hours": "08:00-18:00",
    "analysis.daylight_saving": "none",
    "analysis.aggregation": "floor_area",
    "analysis.moveable_furniture": "include",
    "performance.trace_scope": "all",
    "performance.linked_zones": {},
    "performance.linked_zones.*": [],
    "performance.linked_zones.*[]": "draft",
    "sky.basis": "Reinhart",
    "sky.MF": 4,
    "radiance.ambient": {"draft": "", "full": ""},
    "radiance.ambient.draft": "",
    "radiance.ambient.full": "",
    "radiance.direct": "",
    "blinds.transmittance_diffuse": 0.0,
    "blinds.transmittance_specular": 0.0,
    "blinds.room_side_reflectance": 0.0,
    "blinds.trigger_area": "room",
    "option2.equinox_dates": [],
    "option2.equinox_dates[]": "03-21",
    "option2.clearest_day_window_days": 0,
    "option2.times": [],
    "option2.times[]": "09:00",
    "option2.sky": "weather_file",
    "option2.lux_min": 0.0,
    "option2.lux_max": 0.0,
    "option2.blinds": "exclude",
    "option2.time_basis": "weather_file_local",
    "views.eye_height_m": 1.0,
    "views.eye_height_source": "draft",
    "views.max_interior_glass": VIEWS_DEFAULT_MAX_INTERIOR_GLASS,
    "views.angular_step_deg": VIEWS_DEFAULT_ANGULAR_STEP_DEG,
    "report.renderer": RENDERERS[0],
    "report.deviations": [],
    "report.deviations[]": "draft",
    "report.prepared_by": "draft",
    "atlas.drawings": [],
    "atlas.drawings[]": {
        "storey": "draft", "image": "draft.png",
        "points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                   {"pixel": [100.0, 0.0], "model": [10.0, 0.0]}],
    },
    "atlas.drawings[].storey": "draft",
    "atlas.drawings[].image": "draft.png",
    "atlas.drawings[].opacity": ATLAS_DEFAULT_OPACITY,
    "atlas.drawings[].points": [{"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
                                {"pixel": [100.0, 0.0], "model": [10.0, 0.0]}],
    "atlas.drawings[].points[]": {"pixel": [0.0, 0.0], "model": [0.0, 0.0]},
    "atlas.drawings[].points[].pixel": [0.0, 0.0],
    "atlas.drawings[].points[].pixel[]": 0.0,
    "atlas.drawings[].points[].model": [0.0, 0.0],
    "atlas.drawings[].points[].model[]": 0.0,
})

#: Mappings whose keys are roles, glass classes or storeys: ``*`` in a pattern.
_STAR_MAPPINGS = frozenset({"glazing.frame_share", "glazing.dirt_factor", "performance.linked_zones"})


def _pattern(parent: str, key) -> str:
    """The stand-in pattern of mapping key ``key`` under pattern ``parent``."""
    if not parent:
        return str(key)
    if parent in _STAR_MAPPINGS or (parent == "reflectance" and key != "policy"):
        return f"{parent}.*"
    return f"{parent}.{key}"


def _with_stand_ins(value, pattern: str):
    """A copy of ``value`` with each FILL_ME that has a stand-in replaced by
    a fresh copy of it; a FILL_ME with none is left for the typed load to
    refuse."""
    if isinstance(value, dict):
        return {key: _with_stand_ins(item, _pattern(pattern, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_stand_ins(item, f"{pattern}[]") for item in value]
    if isinstance(value, str) and value.strip() == FILL_ME and pattern in _DRAFT_STAND_INS:
        return copy.deepcopy(_DRAFT_STAND_INS[pattern])
    return value


#: A path in an error text ends at any character outside these.
_NAME_CHAR = "A-Za-z0-9_"

#: The named key right after an ancestor cut, once one leading list index is
#: skipped -- ``pixel`` for ``atlas.drawings[0].points[1].pixel[0]`` cut
#: before ``[1]``. None when nothing but indices follows the cut.
_BRANCH_RE = re.compile(r"^(?:\[\d+\])?\.([A-Za-z_][A-Za-z0-9_]*)")


def _branch(path: str, cut: int) -> str | None:
    match = _BRANCH_RE.match(path[cut:])
    return match.group(1) if match else None


def _names_unfinished(error: str, unfinished: Sequence[str]) -> bool:
    """Whether a check error names an unfinished path (or something inside
    it: ``glazing.frame_share.default`` for ``glazing.frame_share``) or an
    ancestor of one below its section, named itself
    (``atlas.drawings[0].points`` for ``atlas.drawings[0].points[1].pixel``,
    but not ``scene.files`` in ``scene.files[1]``).

    Two different checks can report the identical ancestor text: the atlas
    pixel-spacing and model-spacing checks both open with
    ``atlas.drawings[i].points:`` and differ only in the words after it. An
    ancestor match there must not blind itself to which one it is looking
    at -- an unfinished ``points[j].model...`` may skip only the check that
    itself talks about ``model``, never the sibling check about ``pixel``,
    and the same the other way round. So once an ancestor's text matches,
    the branch named right after the cut (if any) must also appear in the
    error before the match counts.
    """
    for path in unfinished:
        if re.search(rf"(?<![{_NAME_CHAR}.]){re.escape(path)}(?![{_NAME_CHAR}])", error):
            return True
        cuts = [m.start() for m in re.finditer(r"[.\[]", path)]
        for cut in cuts[1:]:
            ancestor = re.escape(path[:cut])
            if not re.search(rf"(?<![{_NAME_CHAR}.]){ancestor}(?![{_NAME_CHAR}.\[])", error):
                continue
            branch = _branch(path, cut)
            if branch is None or branch in error:
                return True
    return False


def check_project_draft(path: str | pathlib.Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check a project.yaml that may still hold FILL_ME (wizard spec 7.3,
    W9). Returns ``(errors, unfinished)``: ``unfinished`` is every path still
    FILL_ME, in file order, as ``load_project`` lists them; ``errors`` is
    every other refusal ``load_project`` would give.

    Each FILL_ME is replaced, in a copy, by a stand-in of the right type
    (``_DRAFT_STAND_INS``) and the copy goes through the same typed load as
    ``load_project``. A typing refusal is the one error returned. Then
    ``check_project`` runs, and an error that names an unfinished path is
    dropped: the check needs a value that is not there yet. Nothing is
    written.
    """
    path = pathlib.Path(path).absolute()
    try:
        raw = yaml.safe_load(read_input_text(path))
    except yaml.YAMLError as exc:
        return (f"project.yaml is not valid YAML: {' '.join(str(exc).split())}",), ()
    if not isinstance(raw, dict):
        return ("project.yaml: project.yaml must be a mapping at the top level",), ()
    unfinished = tuple(_fill_me_paths(raw, ""))
    try:
        project = _typed_project(_with_stand_ins(raw, ""), path.parent)
    except ProjectError as exc:
        return (str(exc),), unfinished
    errors, _ = check_project(project)
    return tuple(e for e in errors if not _names_unfinished(e, unfinished)), unfinished


def load_scene_settings(path: str | pathlib.Path) -> SceneSpec:
    """Read and check only project.yaml's ``scene:`` section (wizard spec
    7.3): the units and the scene files, each of which must exist. The rest
    of the file may still say FILL_ME; the scene may not."""
    path = pathlib.Path(path).absolute()
    raw = yaml.safe_load(read_input_text(path))
    if not isinstance(raw, dict):
        _fail("project.yaml must be a mapping at the top level")
    section = _get_section(raw, "scene", "scene")
    unfinished = _fill_me_paths(section, "scene")
    if unfinished:
        one = len(unfinished) == 1
        _fail(
            f"{', '.join(unfinished)} still {'says' if one else 'say'} {FILL_ME}; set "
            f"{'it' if one else 'them'} in project.yaml"
        )
    scene = _load_scene(raw, path.parent)
    if not scene.files:
        _fail("scene.files must name at least one .rad file")
    errors: list[str] = []
    for i, file in enumerate(scene.files):
        _check_file(errors, f"scene.files[{i}]", file)
    if errors:
        _fail("; ".join(errors))
    return scene
