# leed_mcp/adapters/project_keys.py
"""Every project.yaml key ``inputs(action="update", file="project")`` may
set or remove (spec 6.3).

The loader's key sets are inline literals inside each ``_load_*`` function
of ``project/config.py``, and nothing public lists them, so the table is
written out here. ``tests/mcp/test_mcp_project_keys.py`` holds it against
the loader: every key in it is accepted by ``check_project_draft``, and
every key of the template is in it. Role names come from the vocabularies
``ROLES`` and ``EXTERIOR_GLAZING_ROLES``, glass classes from
``DIRT_CLASSES``. A storey name is one dotted part, written ``*``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from leed_daylight.project.config import DIRT_CLASSES
from leed_daylight.project.materials import EXTERIOR_GLAZING_ROLES, ROLES
from leed_mcp.envelope import refusal
from leed_mcp.errors import Refused

__all__ = ["ALLOWED", "KEYS", "Key", "check_keys", "rule_for"]


@dataclass(frozen=True)
class Key:
    """One key path, and whether ``unset`` may remove it."""

    path: str
    optional: bool = False


def _keys(paths: Iterable[str], optional: bool = False) -> tuple[Key, ...]:
    return tuple(Key(p, optional) for p in paths)


#: ``glazing.frame_share`` keys that name a glass class, not a role
#: (``project/config.py`` ``_FRAME_SHARE_CLASS_KEYS``).
_FRAME_SHARE_CLASSES = ("sloped", "skylight")

KEYS: tuple[Key, ...] = (
    *_keys(["project.name", "project.credit"]),
    *_keys(f"site.{k}" for k in ("latitude_N", "longitude_E", "meridian_radiance", "north_rotation_deg",
                                 "north_rotation_source")),
    *_keys(["scene.units", "scene.files", "standard.daylight", "standard.credit_rules", "weather.file"]),
    *_keys(["glazing.dirt_factor", "glazing.frame_share", "glazing.frame_share.default",
            "glazing.frame_reflectance", "glazing.frame_style", "glazing.reveal_depth_m",
            "glazing.reveal_reflectance", "glazing.vt_includes_dirt"]),
    *_keys((f"glazing.dirt_factor.{c}" for c in DIRT_CLASSES), optional=True),
    *_keys((f"glazing.frame_share.{k}" for k in (*_FRAME_SHARE_CLASSES, *sorted(EXTERIOR_GLAZING_ROLES))),
           optional=True),
    *_keys(["glazing.skylight_well", "glazing.skylight_well_reflectance"], optional=True),
    Key("reflectance.policy"),
    *_keys((f"reflectance.{role}" for role in sorted(ROLES)), optional=True),
    *_keys(["grid.spacing_m", "grid.work_plane_m", "grid.min_clearance_m"]),
    Key("grid.points_file", optional=True),
    *_keys(f"analysis.{k}" for k in ("hours", "daylight_saving", "aggregation", "moveable_furniture")),
    *_keys(["performance.trace_scope", "performance.linked_zones"]),
    Key("performance.linked_zones.*", optional=True),
    *_keys(["sky.basis", "sky.MF", "radiance.ambient", "radiance.ambient.draft", "radiance.ambient.full",
            "radiance.direct"]),
    *_keys(f"blinds.{k}" for k in ("transmittance_diffuse", "transmittance_specular", "room_side_reflectance")),
    Key("blinds.trigger_area", optional=True),
    *_keys(f"option2.{k}" for k in ("equinox_dates", "clearest_day_window_days", "times", "sky", "lux_min",
                                    "lux_max", "blinds", "time_basis")),
    Key("views", optional=True),
    *_keys(["views.eye_height_m", "views.eye_height_source"]),
    *_keys(["views.max_interior_glass", "views.angular_step_deg"], optional=True),
    *_keys(["report", "report.renderer", "report.deviations", "report.prepared_by"], optional=True),
    *_keys(["atlas", "atlas.drawings"], optional=True),
    *_keys(["calculator", "calculator.units"], optional=True),
)

_BY_PATH = {key.path: key for key in KEYS}

#: The table in words, for the tool description and the refusal.
ALLOWED = ("the keys of the project.yaml template; glazing.frame_share.<glazing role, sloped or skylight>; "
           "glazing.dirt_factor.<vertical, sloped or skylight>; glazing.skylight_well and "
           "glazing.skylight_well_reflectance; reflectance.<role>; grid.points_file; blinds.trigger_area; "
           "performance.linked_zones.<storey>; and the optional sections views, report, atlas and calculator")


def rule_for(path: str) -> Key | None:
    """The table's entry for ``path``, a storey name matching ``*``."""
    if path in _BY_PATH:
        return _BY_PATH[path]
    head, _, last = path.rpartition(".")
    if last and head and f"{head}.*" in _BY_PATH:
        return _BY_PATH[f"{head}.*"]
    return None


def check_keys(values: Mapping[str, object], unset: Sequence[str]) -> None:
    """Refuse (``INPUT_INVALID``) a key not in the table, an ``unset`` of a
    key that is not optional, a key both set and removed, and a key given
    with one of its own parents."""
    problems = [f"{k!r} is not a project.yaml key this tool sets" for k in values if rule_for(k) is None]
    for key in unset:
        rule = rule_for(key)
        if rule is None:
            problems.append(f"{key!r} is not a project.yaml key this tool removes")
        elif not rule.optional:
            problems.append(f"{key!r} is required; set it instead of removing it")
    problems += [f"{k!r} is both set and removed" for k in sorted(set(values) & set(unset))]
    given = [*values, *unset]
    problems += [f"{k!r} and its parent {p!r} are both given; give one"
                 for k in given for p in given if k.startswith(f"{p}.")]
    if problems:
        raise Refused(refusal("INPUT_INVALID", "; ".join(problems) + f". Allowed: {ALLOWED}.", field="values"))
