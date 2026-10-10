# leed_daylight/deliver/atlas/page.py
"""The atlas page: one self-contained ``atlas.html`` (atlas spec 8, 9, 13).

Generated fresh from the case and the bundle; the author's earlier page is
not ported. The page holds:

* a header: project, case, date, tool version and, per current option, the
  project result and points exactly as its results file states them;
* every option that is not shown, with the reason ``load_case`` gave;
* one tab per storey: a layer switch, one plan SVG per layer (``plan_svgs``:
  ``PlanSpec`` -> ``plan_canvas`` -> ``svgtext.to_svg``, as the Word reports
  draw them) and a room table; a storey's drawing is held once in the page
  (``drawings_svg``) and every plan of that storey draws it by reference;
  a storey with a split or merged space lists its sentence under the table
  (merge-split spec 7);
* a 3D tab: the element ``#atlas-3d`` the inlined viewer draws into, fed by
  ``window.__ATLAS__`` (``payload``).

The page's DOM contract with ``web/viewer.js`` is exactly that element and
that payload. ``web/page.js`` switches tabs and plan layers. No part of the
page is fetched: the viewer, its stylesheet, every plan and every drawing
are inlined, and no font is loaded.
"""
from __future__ import annotations

import base64
import html
import json
import pathlib
import re
from collections.abc import Mapping

import numpy as np

from ...constants import TOOL_VERSION
from ...project.spaces import space_of
from ..data import CaseData, DeliverError
from ..figures.spec import FAIL_RGB, NEUTRAL_RGB, PASS_RGB, PlanPolygon, PlanSpec, Underlay, plan_canvas, ramp
from ..figures.svgtext import shared_images_svg, to_svg
from .bundle import Bundle, PointSet, grid_points
from .colours import DEFAULT_OFF, KIND_RGBA, ROOM_FILL_RGB
from .layers import Layer

WEB = pathlib.Path(__file__).resolve().parent / "web"
#: Spec 13: the page warns above this size; there is no hard limit.
SIZE_WARNING_BYTES = 25_000_000
#: Positions travel as int16: the model's longest side spans this many steps.
_QUANT_STEPS = 32000.0
#: Where the legend samples the colour ramp (every stop of the ramp is one).
_RAMP_AT = tuple(i / 10.0 for i in range(11))
_MAX_STOREYS = 127  # int8 per vertex
_MAX_ROOMS = 65535  # uint16 per point
_OPTION_TITLES = {
    "option1": "Option 1, annual simulation",
    "option2": "Option 2, illuminance calculation",
    "views": "Quality Views",
}


# -- geometry ------------------------------------------------------------------

def _earclip(p2: np.ndarray) -> list[tuple[int, int, int]]:
    """Index triples covering a simple polygon given as (n, 2). Ported
    (PORTED.md): ear clipping, so a concave outline (a wall with a
    window hole drawn as one polygon) is never folded as a fan would."""
    n = len(p2)
    if n < 3:
        return []
    idx = list(range(n))
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += p2[i, 0] * p2[j, 1] - p2[j, 0] * p2[i, 1]
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return ((p2[a, 0] - p2[o, 0]) * (p2[b, 1] - p2[o, 1])
                - (p2[a, 1] - p2[o, 1]) * (p2[b, 0] - p2[o, 0]))

    def inside(a, b, c, q):
        d1, d2, d3 = cross(a, b, q), cross(b, c, q), cross(c, a, q)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    out, guard = [], 0
    while len(idx) > 3 and guard < 4 * n + 16:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) <= 0:
                continue  # reflex or degenerate
            if any(inside(a, b, c, q) for q in idx if q not in (a, b, c)):
                continue
            out.append((a, b, c))
            idx.pop(k)
            guard = 0
            break
        else:
            break  # nothing clipped: stop
    if len(idx) == 3:
        out.append(tuple(idx))
    if not out:  # last resort, a fan
        out = [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return out


def _triangulate(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) polygon -> its (m, 3, 3) triangles and its unit normal."""
    normal = np.zeros(3)
    for i in range(len(v)):
        normal += np.cross(v[i], v[(i + 1) % len(v)])
    length = np.linalg.norm(normal)
    normal = normal / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    p2 = v[:, keep]
    if normal[drop] < 0:
        p2 = p2[:, ::-1]
    tris = [v[list(t)] for t in _earclip(np.ascontiguousarray(p2))]
    return (np.array(tris, dtype=np.float64) if tris else np.zeros((0, 3, 3))), normal


# -- payload -------------------------------------------------------------------

class _Blob:
    """One byte buffer with a name -> [byte offset, element count] index;
    every part starts on a 4-byte boundary, as typed arrays need."""

    def __init__(self):
        self.buf = bytearray()
        self.parts: dict[str, list[int]] = {}

    def add(self, name: str, array: np.ndarray) -> None:
        while len(self.buf) % 4:
            self.buf.append(0)
        array = np.ascontiguousarray(array)
        self.parts[name] = [len(self.buf), int(array.size)]
        self.buf.extend(array.tobytes())


def _shell(bundle: Bundle) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per kind: triangle corners (K, 3), normals (K, 3) and storeys (K,)."""
    out = {}
    for kind, surfaces in bundle.surfaces.items():
        corners, normals, storeys = [], [], []
        start = 0
        for count, storey in zip(surfaces.counts.tolist(), surfaces.storey.tolist()):
            tris, normal = _triangulate(surfaces.verts[start:start + count])
            start += count
            if not len(tris):
                continue
            corners.append(tris.reshape(-1, 3))
            normals.append(np.tile(normal, (tris.shape[0] * 3, 1)))
            storeys.append(np.full(tris.shape[0] * 3, storey))
        if corners:
            out[kind] = (np.vstack(corners), np.vstack(normals), np.concatenate(storeys))
    return out


def _legend(layer: Layer) -> dict:
    if layer.scale is None:
        return {"items": [[layer.legend[0], list(PASS_RGB)], [layer.legend[1], list(FAIL_RGB)]]}
    scale = layer.scale
    return {
        "ramp": [[t, list(ramp(t))] for t in _RAMP_AT],
        "ticks": [[scale.fraction(t), f"{t:g}"] for t in scale.ticks()],
        "marks": [scale.fraction(m) for m in scale.marks],
    }


def payload(bundle: Bundle) -> dict:
    """``window.__ATLAS__`` (atlas plan, interface contract): one base64
    byte buffer and its parts, the kinds, storeys, rooms and layers. Layer 0
    is the model grid (parts ``grid_*``); layer i + 1 is the bundle's layer i
    (parts ``l<i>_*``); each layer names its parts in ``part``."""
    if len(bundle.storeys) > _MAX_STOREYS:
        raise DeliverError(f"the 3D view holds at most {_MAX_STOREYS} storeys; this case has {len(bundle.storeys)}")
    if len(bundle.rooms) > _MAX_ROOMS:
        raise DeliverError(f"the 3D view holds at most {_MAX_ROOMS} rooms; this case has {len(bundle.rooms)}")
    shell = _shell(bundle)
    point_sets: list[tuple[str, PointSet]] = [("grid", bundle.grid)]
    point_sets += [(f"l{i}", points) for i, (_, points) in enumerate(bundle.layers)]
    everything = [corners for corners, _, _ in shell.values()] + [points.xyz for _, points in point_sets]
    everything = [xyz for xyz in everything if xyz.size]
    if everything:
        allxyz = np.vstack(everything)
        lo = allxyz.min(axis=0)
        span = float(np.max(allxyz.max(axis=0) - lo))
    else:
        lo, span = np.zeros(3), 0.0
    step = span / _QUANT_STEPS if span > 0.0 else 1.0

    def quantise(xyz: np.ndarray) -> np.ndarray:
        return np.round((np.asarray(xyz, dtype=float).reshape(-1, 3) - lo) / step).astype("<i2")

    blob = _Blob()
    kinds = []
    for kind, (corners, normals, storeys) in shell.items():
        blob.add(f"k_{kind}_pos", quantise(corners))
        blob.add(f"k_{kind}_nrm", np.clip(np.round(normals * 127.0), -127, 127).astype("i1"))
        blob.add(f"k_{kind}_sto", storeys.astype("i1"))
        kinds.append({"k": kind, "rgba": list(KIND_RGBA[kind]), "on": kind not in DEFAULT_OFF})
    room_index = {room.name: i for i, room in enumerate(bundle.rooms)}
    for name, points in point_sets:
        blob.add(f"{name}_pos", quantise(points.xyz))
        blob.add(f"{name}_sto", points.storey.astype("i1"))
        blob.add(f"{name}_room", np.array([room_index[r] for r in points.room], dtype="<u2"))
        blob.add(f"{name}_rgb", points.rgb.astype("u1").reshape(-1))
        blob.add(f"{name}_val", points.values.astype("<f4"))
    layers = [{"key": "model", "title": "Model", "unit": "", "pass_fail": False, "fail_when": None, "part": "grid",
               "legend": {"items": [["grid point", list(NEUTRAL_RGB)]]}}]
    layers += [
        {"key": layer.key, "title": layer.title, "unit": layer.scale.unit if layer.scale else "",
         "pass_fail": layer.scale is None, "fail_when": layer.fail_when, "part": f"l{i}", "legend": _legend(layer)}
        for i, (layer, _) in enumerate(bundle.layers)
    ]
    return {
        "blob": base64.b64encode(bytes(blob.buf)).decode("ascii"),
        "parts": blob.parts,
        "quant": [float(lo[0]), float(lo[1]), float(lo[2]), float(step)],
        "kinds": kinds,
        "storeys": [{"label": label, "level": level} for label, level in zip(bundle.storeys, bundle.levels)],
        "rooms": [room.name for room in bundle.rooms],
        "layers": layers,
        "cell": bundle.cell_m,
        "work": bundle.work_plane_m,
    }


# -- plans ---------------------------------------------------------------------

def drawing_id(storey_index: int) -> str:
    """The page id of storey ``storey_index``'s drawing."""
    return f"dwg-s{storey_index}"


def drawings_svg(bundle: Bundle, drawings: Mapping[str, Underlay]) -> str:
    """Final review I2: every storey's drawing, once, in one hidden SVG the
    plans of that storey draw by reference (``""`` with no drawing)."""
    return shared_images_svg({
        drawing_id(si): drawings[label] for si, label in enumerate(bundle.storeys) if label in drawings
    })


def plan_svgs(case: CaseData, bundle: Bundle, drawings: Mapping[str, Underlay]) -> dict[str, dict[str, str]]:
    """Storey label, then layer key (``"model"`` first), to the plan SVG.
    Included rooms are filled, excluded rooms grey and hatched; the model
    grid is drawn as dots, a result layer as coloured cells; window groups
    as lines and roof glass as dashed outlines; split lines dashed, each
    once (merge-split spec 7: the line two halves share is one array); a
    space of several pieces has every piece drawn and one label, where
    ``StoreyPlan.centroids`` puts it; a storey's drawing, when
    given, lies under every one of its plans, over the room fills. A
    plan draws the drawing by reference to ``drawing_id`` (final review
    I2): the page holds its bytes once, in ``drawings_svg``. Every SVG has
    its own id prefix, since the page holds them all."""
    layer_sets: list[tuple[str, Layer | None, PointSet]] = [("model", None, bundle.grid)]
    layer_sets += [(layer.key, layer, points) for layer, points in bundle.layers]
    out: dict[str, dict[str, str]] = {}
    for si, label in enumerate(bundle.storeys):
        plan = case.storey(label)
        rooms = [room for room in bundle.rooms if room.name in plan.rooms]
        outlines = tuple(ring for room in rooms if room.included for ring in plan.outlines.get(room.name, ()))
        excluded = tuple(ring for room in rooms if not room.included for ring in plan.outlines.get(room.name, ()))
        labels = tuple((room.name, *plan.centroids[room.name]) for room in rooms)
        shared = {id(line): line for room in rooms for line in plan.split_lines.get(room.name, ())}
        split_lines = tuple(shared.values())
        here = [g for g in case.groups if g.storey_label == label]
        segments = tuple((g.group_id, seg) for g in here for seg in g.segments)
        polygons = tuple(PlanPolygon(points=outline, dashed=True, label=g.group_id if i == 0 else "")
                         for g in here for i, outline in enumerate(g.outlines))
        underlay = drawings.get(label)
        image_ids = {underlay.data: drawing_id(si)} if underlay is not None else None
        svgs: dict[str, str] = {}
        for li, (key, layer, points) in enumerate(layer_sets):
            mask = points.storey == si
            values = points.values[mask]
            spec = PlanSpec(
                title=f"Storey {label}: {layer.title if layer else 'model'}",
                outlines=outlines, labels=labels, points_xy=points.xyz[mask, :2], cell_m=bundle.cell_m,
                north_rotation_deg=case.project.site.north_rotation_deg,
                values=values if layer is not None and layer.scale is not None else None,
                scale=layer.scale if layer is not None else None,
                passed=(values != 0.0) != layer.fail_when if layer is not None and layer.fail_when is not None else None,
                segments=segments, polygons=polygons,
                legend_labels=layer.legend if layer is not None else ("passes", "fails"),
                underlay=underlay, excluded_outlines=excluded, room_fill=ROOM_FILL_RGB,
                point_style="dot" if layer is None else "cell", split_lines=split_lines,
            )
            svgs[key] = to_svg(plan_canvas(spec)[0], id_prefix=f"p{si}-{li}-", image_ids=image_ids)
        out[label] = svgs
    return out


# -- html ----------------------------------------------------------------------

def _e(text) -> str:
    return html.escape(str(text))


def _pct(value) -> str:
    return "not computed" if value is None else f"{float(value):.1f} %"


def _score(score: Mapping) -> str:
    """The points as the results file states them (as the Word reports word them)."""
    if score.get("points") is None:
        return f"not awarded ({score.get('reason') or 'no reason recorded'})"
    return f"{score['points']} ({score.get('band') or 'below the lowest band'})"


def _result_line(option: str, results: Mapping) -> str:
    if option == "option1":
        by_points = results.get("headline") == "points"
        sda = results["sda_point_pct"] if by_points else results["sda_area_pct"]
        ase = results["ase_point_pct"] if by_points else results["ase_area_pct"]
        weighting = "point weighted" if by_points else "floor-area weighted"
        return f"sDA {_pct(sda)} ({weighting}), ASE {_pct(ase)}"
    if option == "option2":
        return f"{_pct(results['area_pct'])} of floor area in the band"
    return f"{_pct(results['quality_area_pct'])} of floor area with a Quality View"


def _in_project(text: str, folder: pathlib.Path) -> str:
    """``text`` with every path inside the project ``folder`` written
    relative to it (final review M3): the page travels in the package, so
    it names no path of the machine that built it. ``load_case``'s own
    reasons keep their paths."""
    forms = {str(folder), str(folder.absolute()), str(folder.resolve())} - {"", "."}
    for form in sorted(forms, key=len, reverse=True):
        text = re.sub(r"(?<![\w.~/\\-])" + re.escape(form) + r"[/\\]", "", text)
    return text


def _header(case: CaseData) -> str:
    parts = [
        '<header class="atlas-head">',
        f"<h1>{_e(case.project.name)}</h1>",
        f'<p class="meta">Floor atlas, case {_e(case.case)} · {_e(case.created.strftime("%d %B %Y"))} · '
        f"leed-daylight {_e(TOOL_VERSION)}</p>",
    ]
    if case.options:
        rows = "".join(
            f"<tr><td>{_e(_OPTION_TITLES[option])}</td><td>{_e(_result_line(option, data.results))}</td>"
            f"<td>{_e(_score(data.results.get('score', {})))}</td></tr>"
            for option, data in case.options.items()
        )
        parts.append('<table class="results"><thead><tr><th>Credit option</th><th>Project result</th>'
                     f"<th>Points</th></tr></thead><tbody>{rows}</tbody></table>")
    else:
        parts.append('<p class="check">Model check: no current results, so the plans show the model and its '
                     "grid only.</p>")
    if case.skipped:
        items = "".join(f"<li>{_e(_in_project(reason, case.folder))}</li>" for reason in case.skipped.values())
        parts.append(f'<section class="skipped"><h2>Results not shown</h2><ul>{items}</ul></section>')
    if case.warnings:
        items = "".join(f"<li>{_e(_in_project(warning, case.folder))}</li>" for warning in case.warnings)
        parts.append(f'<section class="warnings"><h2>Warnings</h2><ul>{items}</ul></section>')
    parts.append("</header>")
    return "\n".join(parts)


_RESULT_COLUMNS = (
    ("option1", "sda_pct", "Option 1 sDA %"),
    ("option1", "ase_pct", "Option 1 ASE %"),
    ("option2", "pass_pct", "Option 2 in band %"),
    ("views", "quality_pct", "Quality View %"),
)


def _room_table(case: CaseData, bundle: Bundle, label: str) -> str:
    plan = case.storey(label)
    columns = [(option, key, title) for option, key, title in _RESULT_COLUMNS if option in case.options]
    by_room = {
        option: {space["room"]: space for space in case.options[option].results.get("spaces", [])}
        for option in {option for option, _, _ in columns}
    }
    # Open point 11: a space's area is the sum of its rows' area_m2, so a
    # one-piece space shows its own row's number.
    owner = space_of(case.spaces)
    area = {room: sum(float(s.area_m2) for s in case.spaces if owner[s.room] == room) for room in plan.rooms}
    included = {room.name: room.included for room in bundle.rooms}
    head = ["Room", "Area m²", "Points", "Analysis", "Window groups"] + [title for _, _, title in columns]
    rows = []
    for room in plan.rooms:
        groups = ", ".join(g.group_id for g in case.groups if g.room == room) or "none"
        cells = [
            f"<td>{_e(room)}</td>",
            f'<td class="num">{area[room]:.1f}</td>',
            f'<td class="num">{grid_points(bundle, room)}</td>',
            f"<td>{'included' if included.get(room) else 'excluded'}</td>",
            f"<td>{_e(groups)}</td>",
        ]
        for option, key, _ in columns:
            space = by_room[option].get(room)
            cells.append(f'<td class="num">{"" if space is None else f"{float(space[key]):.1f}"}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="rooms"><thead><tr>' + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _layout_notes(case: CaseData, bundle: Bundle, label: str) -> str:
    """The sentence of each split or merged space of the storey, in table
    order, as a list; ``""`` when there is none. The blind clause is stated
    only for a space the atlas shows Option 1 results of (controller
    ruling): without Option 1 results, or for a space Option 1 did not
    analyse, no blind is shown, as in the Option 2 and Quality Views
    reports. A space with no grid point in this case (``grid_points``, as
    the room table counts them: excluded from the case, or included but not
    regularly occupied) states neither the grid clause nor the blind clause
    (controller ruling, pre-flight F2b; review I1); a split space's
    sentence, which states neither, is unaffected."""
    option1 = case.options.get("option1")
    blinds = {space["room"] for space in option1.results.get("spaces", [])} if option1 is not None else set()
    items = ""
    for room in case.storey(label).rooms:
        if grid_points(bundle, room) == 0:
            notes = case.layout_notes_no_grid
        elif room in blinds:
            notes = case.layout_notes
        else:
            notes = case.layout_notes_no_blinds
        if room in notes:
            items += f"<li>{_e(notes[room])}</li>"
    return f'<ul class="layout-notes">{items}</ul>' if items else ""


def _storey_panel(case: CaseData, bundle: Bundle, si: int, label: str, svgs: Mapping[str, str],
                  titles: Mapping[str, str]) -> str:
    options = "".join(f'<option value="{_e(key)}">{_e(titles[key])}</option>' for key in svgs)
    figures = "".join(
        f'<figure class="plan" data-layer="{_e(key)}"{"" if i == 0 else " hidden"}>{svg}</figure>'
        for i, (key, svg) in enumerate(svgs.items())
    )
    hidden = "" if si == 0 else " hidden"
    return (
        f'<section class="panel" data-panel="s{si}" role="tabpanel"{hidden}>'
        f"<h2>Storey {_e(label)}</h2>"
        f'<label class="layer-switch">Plan layer <select data-plan-select="s{si}">{options}</select></label>'
        f'<div class="plans">{figures}</div>'
        f"{_room_table(case, bundle, label)}"
        f"{_layout_notes(case, bundle, label)}"
        "</section>"
    )


def _script_json(value) -> str:
    """JSON safe inside a <script> element: no "<" is left, so neither "</" nor "<!--" can occur."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def write_page(path, case: CaseData, bundle: Bundle, drawings: Mapping[str, Underlay]) -> int:
    """Write ``atlas.html`` to ``path``; the number of bytes written."""
    svgs = plan_svgs(case, bundle, drawings)
    titles = {"model": "Model"} | {layer.key: layer.title for layer, _ in bundle.layers}
    web = {name: (WEB / name).read_text(encoding="utf-8") for name in ("page.css", "viewer.css", "page.js",
                                                                       "viewer.js")}
    tabs = "".join(
        f'<button type="button" role="tab" data-tab="s{si}" aria-selected="{"true" if si == 0 else "false"}">'
        f"Storey {_e(label)}</button>"
        for si, label in enumerate(bundle.storeys)
    ) + '<button type="button" role="tab" data-tab="3d" aria-selected="false">3D</button>'
    panels = "".join(
        _storey_panel(case, bundle, si, label, svgs[label], titles) for si, label in enumerate(bundle.storeys)
    )
    three_d = (
        '<section class="panel" data-panel="3d" role="tabpanel" hidden>'
        "<h2>3D</h2>"
        '<p class="meta">The prepared model as the simulation sees it, and one tile per point of the chosen '
        "layer.</p>"
        '<div id="atlas-3d" class="v3d-slot"></div>'
        "</section>"
    )
    title = f"{case.project.name}, case {case.case}: floor atlas"
    document = "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title>",
        f"<style>\n{web['page.css']}\n{web['viewer.css']}\n</style>",
        "</head>",
        "<body>",
        _header(case),
        f'<nav class="tabs" role="tablist">{tabs}</nav>',
        drawings_svg(bundle, drawings),
        f"<main>{panels}{three_d}</main>",
        f"<script>window.__ATLAS__={_script_json(payload(bundle))};</script>",
        f"<script>\n{web['page.js']}\n</script>",
        f"<script>\n{web['viewer.js']}\n</script>",
        "</body>",
        "</html>",
        "",
    ])
    data = document.encode("utf-8")
    pathlib.Path(path).write_bytes(data)
    return len(data)


def size_warning(n_bytes: int) -> str | None:
    """Spec 13: a warning above 25 MB, else ``None``."""
    if n_bytes <= SIZE_WARNING_BYTES:
        return None
    return f"atlas.html is {n_bytes / 1e6:.1f} MB, above 25 MB; a browser may be slow to open it"
