# tests/test_atlas_page.py
"""The atlas page (atlas spec 8, 9, 13, 15).

One self-contained file: a tab per storey with its plans (one SVG per
layer, through ``plan_canvas`` and the pure SVG writer) and its room table,
a 3D tab the inlined viewer draws into, a header with the results files'
own numbers, every option that is not shown with its reason, and the
``window.__ATLAS__`` payload. No test opens a browser; the user checks the
page by eye once. Every number is invented and round.
"""
from __future__ import annotations

import base64
import json
import re

import numpy as np
import pytest
import yaml

from atlas_folder import atlas_inputs
from deliver_folder import build_deliver_project, build_two_storey_project, edit_results
from leed_daylight.constants import TOOL_VERSION
from leed_daylight.deliver.atlas import page
from leed_daylight.deliver.atlas.bundle import Bundle, PointSet, RoomLabel, SurfaceSet
from leed_daylight.deliver.atlas.colours import DEFAULT_OFF, KIND_RGBA, ROOM_FILL_RGB
from leed_daylight.deliver.data import DeliverError
from leed_daylight.deliver.figures.spec import NEUTRAL_RGB, Underlay, rgb_hex

WEB = page.WEB
URL_ATTRIBUTE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']*)["']""", re.I)
URL_FUNCTION = re.compile(r"""url\(\s*["']?([^"')]*)""", re.I)
URL_IMPORT = re.compile(r"""@import\s+(?:url\()?\s*["']?([^"');\s]*)""", re.I)


def _write(tmp_path, case, bundle, drawings=None) -> str:
    out = tmp_path / "atlas"
    out.mkdir()
    n = page.write_page(out / "atlas.html", case, bundle, drawings or {})
    assert sorted(p.name for p in out.iterdir()) == ["atlas.html"]
    data = (out / "atlas.html").read_bytes()
    assert n == len(data)
    return data.decode("utf-8")


def _payload(html: str) -> dict:
    match = re.search(r"<script>window\.__ATLAS__=(.*?);</script>", html, re.S)
    assert match, "no payload"
    return json.loads(match.group(1).replace("<\\/", "</"))


def _part(payload: dict, name: str, dtype: str) -> np.ndarray:
    blob = base64.b64decode(payload["blob"])
    offset, count = payload["parts"][name]
    return np.frombuffer(blob, dtype=dtype, count=count, offset=offset)


def _header(html: str) -> str:
    return html.split("<nav", 1)[0]


# -- one file, no network ------------------------------------------------------

def test_the_page_is_one_file_with_no_url(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    underlay = Underlay(b"\x89PNG not decoded here", "image/png", 10, 10, (0.5, 0.0, 0.0, -0.5, 0.0, 5.0), 0.35)

    html = _write(tmp_path, case, bundle, {"L0": underlay})

    found = URL_ATTRIBUTE.findall(html) + URL_FUNCTION.findall(html) + URL_IMPORT.findall(html)
    assert found, "the scan found nothing to check"
    for value in found:
        assert not value.lower().startswith(("http:", "https:", "//")), value
        assert value.startswith(("#", "data:")), value
    assert "http:" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https:" not in html
    assert "@font-face" not in html
    assert html.startswith("<!DOCTYPE html>")


def test_one_tab_per_storey_and_the_3d_tab(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=())
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    html = _write(tmp_path, case, bundle)

    tabs = re.findall(r'<button type="button" role="tab" data-tab="([^"]+)"[^>]*>([^<]*)</button>', html)
    assert tabs == [("s0", "Storey L0"), ("s1", "Storey L1"), ("3d", "3D")]
    assert re.findall(r'data-panel="([^"]+)"', html) == ["s0", "s1", "3d"]


def test_the_page_provides_the_viewers_dom_and_inlines_it(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    html = _write(tmp_path, case, bundle)

    viewer_js = (WEB / "viewer.js").read_text(encoding="utf-8")
    for element in set(re.findall(r'getElementById\("([^"]+)"\)', viewer_js)):
        assert html.count(f'id="{element}"') == 1
    assert viewer_js in html
    assert (WEB / "viewer.css").read_text(encoding="utf-8") in html
    assert (WEB / "page.js").read_text(encoding="utf-8") in html
    assert html.index("window.__ATLAS__=") < html.index(viewer_js)
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "an id is used twice"


# -- header and missing layers ---------------------------------------------------

def test_the_header_numbers_are_the_results_files(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea)
    edit_results(folder, "option1", lambda p: p.update(sda_area_pct=61.0, ase_area_pct=7.0))
    edit_results(folder, "option2", lambda p: p.update(area_pct=52.0))
    edit_results(folder, "views", lambda p: p.update(quality_area_pct=83.0))
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    header = _header(_write(tmp_path, case, bundle))

    assert "Example Tower" in header
    assert "case base_case" in header
    assert case.created.strftime("%d %B %Y") in header
    assert f"leed-daylight {TOOL_VERSION}" in header
    assert "sDA 61.0 % (floor-area weighted), ASE 7.0 %" in header
    assert "52.0 % of floor area in the band" in header
    assert "83.0 % of floor area with a Quality View" in header
    assert "1 (40.0 % or more)" in header  # option1 points, as results.json states them
    assert "not awarded (below the 75 % threshold)" in header  # views


def test_a_stale_option_is_listed_with_its_reason(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    raw = yaml.safe_load((folder / "project.yaml").read_text())
    raw["glazing"]["dirt_factor"] = 0.8
    (folder / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), newline="\n")
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    header = _header(_write(tmp_path, case, bundle))

    assert "option1 results for case &#x27;base_case&#x27; are out of date: project.yaml changed" in header
    assert "no option2 results for case &#x27;base_case&#x27;" in header
    assert "Model check" in header


def test_the_page_names_no_path_of_the_project_folder(tmp_path, write_example, write_wea):
    # Final review M3: load_case's reasons hold absolute paths; the page,
    # which a package carries, shows them relative to the project folder.
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=("option1",))
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    assert str(folder.resolve()) in case.skipped["option2"] or str(folder) in case.skipped["option2"]

    html = _write(tmp_path, case, bundle)

    for form in {str(folder), str(folder.absolute()), str(folder.resolve())}:
        assert form not in html
    header = _header(html)
    assert f"(_run/option2/{case.case}/option2_results.json not found)" in header
    assert f"(_run/views/{case.case}/views_results.json not found)" in header


def test_a_model_check_lists_every_option_and_has_only_the_model_layer(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    html = _write(tmp_path, case, bundle)

    header = _header(html)
    for option in ("option1", "option2", "views"):
        assert f"no {option} results for case" in header
    assert re.findall(r'<option value="([^"]+)"', html) == ["model"]
    assert [layer["key"] for layer in _payload(html)["layers"]] == ["model"]


# -- plans -----------------------------------------------------------------------

def test_plans_per_storey_and_layer(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    svgs = page.plan_svgs(case, bundle, {})

    keys = ["model"] + [layer.key for layer, _ in bundle.layers]
    assert list(svgs) == ["L0", "L1"]
    assert all(list(by_layer) == keys for by_layer in svgs.values())
    model = svgs["L0"]["model"]
    assert model.startswith("<svg") and "Storey L0: model" in model
    assert model.count(f'fill="{rgb_hex(NEUTRAL_RGB)}"/>') >= 54  # one dot per grid point of Room North
    assert "<circle" in model
    assert f'fill="{rgb_hex(ROOM_FILL_RGB)}"' in model  # Room North, included
    assert 'fill="url(#p0-0-hatch-' in model  # Store East, excluded
    assert ">Store East</text>" in model and ">Room_North_180</text>" in model
    assert "<circle" not in svgs["L0"]["option1.da_pct"]
    assert "Storey L1: Quality View" in svgs["L1"]["views.quality"]


def test_the_drawing_lies_under_its_storey_only(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=("option1",))
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    underlay = Underlay(b"\x89PNG not decoded here", "image/png", 10, 10, (0.5, 0.0, 0.0, -0.5, 0.0, 5.0), 0.35)

    svgs = page.plan_svgs(case, bundle, {"L1": underlay})

    # Final review I2: each plan draws the storey's drawing by reference;
    # the page holds its bytes once (test_the_drawing_is_embedded_once_per_storey).
    assert all('<use href="#dwg-s1" ' in svg for svg in svgs["L1"].values())
    assert not any("base64" in svg for svg in svgs["L1"].values())
    assert not any("<image" in svg or "<use" in svg for svg in svgs["L0"].values())
    assert 'clip-path="url(#p1-0-clip-0)"' in svgs["L1"]["model"]
    assert 'clip-path="url(#p1-1-clip-0)"' in svgs["L1"]["option1.da_pct"]


def test_the_drawing_is_embedded_once_per_storey(tmp_path, write_example, write_wea):
    # Final review I2: not once per plan layer.
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    underlay = Underlay(b"\x89PNG an invented drawing", "image/png", 10, 10, (0.5, 0.0, 0.0, -0.5, 0.0, 5.0), 0.35)
    assert len(bundle.layers) >= 2

    html = _write(tmp_path, case, bundle, {"L1": underlay})

    data = base64.b64encode(underlay.data).decode("ascii")
    assert html.count(data) == 1
    assert html.count('id="dwg-s1"') == 1
    panel = html.split('data-panel="s1"', 1)[1].split('data-panel="3d"', 1)[0]
    plans = re.findall(r'<figure class="plan"[^>]*>(.*?)</figure>', panel, re.S)
    assert len(plans) == 1 + len(bundle.layers)
    assert all(plan.count('<use href="#dwg-s1" ') == 1 for plan in plans)
    other = html.split('data-panel="s0"', 1)[1].split('data-panel="s1"', 1)[0]
    assert "<use" not in other and "dwg-s0" not in html


def test_the_drawing_lies_over_the_room_fills(tmp_path, write_example, write_wea):
    # Final review I1: the opaque room fill hid the drawing inside every
    # included room; the fills now come first, then the drawing.
    folder = build_two_storey_project(tmp_path, write_example, write_wea, options=("option1",))
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")
    underlay = Underlay(b"\x89PNG not decoded here", "image/png", 10, 10, (0.5, 0.0, 0.0, -0.5, 0.0, 5.0), 0.35)

    svgs = page.plan_svgs(case, bundle, {"L0": underlay})

    fill = f'fill="{rgb_hex(ROOM_FILL_RGB)}"'
    for svg in svgs["L0"].values():
        drawing = re.search(r"<(?:image|use)\b", svg).start()
        assert fill in svg
        assert svg.rindex(fill) < drawing


# -- room tables -------------------------------------------------------------------

def test_the_room_table(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    edit_results(folder, "option1", lambda p: p["spaces"][0].update(sda_pct=62.0, ase_pct=4.0))
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    html = _write(tmp_path, case, bundle)

    panel = html.split('data-panel="s0"', 1)[1].split('data-panel="s1"', 1)[0]
    table = panel.split('<table class="rooms">', 1)[1].split("</table>", 1)[0]
    head = re.findall(r"<th>([^<]*)</th>", table)
    assert head == ["Room", "Area m²", "Points", "Analysis", "Window groups", "Option 1 sDA %", "Option 1 ASE %",
                    "Option 2 in band %", "Quality View %"]
    rows = [re.findall(r"<td[^>]*>([^<]*)</td>", row) for row in re.findall(r"<tr>(.*?)</tr>", table)[1:]]
    north_points = sum(1 for room in bundle.grid.room if room == "Room North")
    assert rows[0][:5] == ["Room North", "24.0", str(north_points), "included", "Room_North_180"]
    assert rows[0][5:7] == ["62.0", "4.0"]
    assert rows[1][:5] == ["Store East", "16.0", "0", "excluded", "none"]
    assert rows[1][5:] == ["", "", "", ""]


# -- payload -------------------------------------------------------------------------

def test_the_payload_matches_the_bundle(tmp_path, write_example, write_wea):
    folder = build_two_storey_project(tmp_path, write_example, write_wea)
    case, _, bundle = atlas_inputs(folder, tmp_path / "work")

    payload = _payload(_write(tmp_path, case, bundle))

    assert set(payload) == {"blob", "parts", "quant", "kinds", "storeys", "rooms", "layers", "cell", "work"}
    assert all(offset % 4 == 0 for offset, _ in payload["parts"].values())
    assert payload["storeys"] == [{"label": "L0", "level": 0.0}, {"label": "L1", "level": 4.0}]
    assert payload["rooms"] == ["Room North", "Store East", "Room Upper"]
    assert payload["cell"] == bundle.cell_m and payload["work"] == bundle.work_plane_m
    assert [k["k"] for k in payload["kinds"]] == list(bundle.surfaces)
    for entry in payload["kinds"]:
        assert entry["rgba"] == list(KIND_RGBA[entry["k"]])
        assert entry["on"] == (entry["k"] not in DEFAULT_OFF)
    ox, oy, oz, scale = payload["quant"]
    grid = _part(payload, "grid_pos", "<i2").reshape(-1, 3) * scale + np.array([ox, oy, oz])
    np.testing.assert_allclose(grid, bundle.grid.xyz, atol=scale / 2 + 1e-12)
    assert _part(payload, "grid_sto", "i1").tolist() == bundle.grid.storey.tolist()
    rooms = _part(payload, "grid_room", "<u2")
    assert [payload["rooms"][i] for i in rooms] == list(bundle.grid.room)
    layers = payload["layers"]
    assert layers[0] == {"key": "model", "title": "Model", "unit": "", "pass_fail": False, "fail_when": None,
                         "part": "grid", "legend": {"items": [["grid point", list(NEUTRAL_RGB)]]}}
    assert [layer["key"] for layer in layers[1:]] == [layer.key for layer, _ in bundle.layers]
    da = layers[1]
    assert (da["part"], da["unit"], da["pass_fail"]) == ("l0", "% of occupied hours at 300 lux or more", False)
    assert da["legend"]["marks"] == [0.5]
    assert da["legend"]["ticks"][0] == [0.0, "0"] and da["legend"]["ticks"][-1] == [1.0, "100"]
    assert len(da["legend"]["ramp"]) == 11
    sda_pass = layers[2]
    assert (sda_pass["pass_fail"], sda_pass["fail_when"]) == (True, False)
    assert [item[0] for item in sda_pass["legend"]["items"]] == ["passes", "fails"]
    _, points = bundle.layers[0]
    assert _part(payload, "l0_val", "<f4").tolist() == points.values.astype(np.float32).tolist()
    assert _part(payload, "l0_rgb", "u1").reshape(-1, 3).tolist() == points.rgb.tolist()
    floor = _part(payload, "k_floor_pos", "<i2")
    assert floor.shape == (3 * 3 * 2 * 3,)  # three quads, two triangles each, three corners, x y z
    assert _part(payload, "k_floor_sto", "i1").tolist() == [0] * 12 + [1] * 6


def _hand_bundle(storeys=("S0",), rooms=("A",), floor=None) -> Bundle:
    grid = PointSet(np.array([[1.0, 1.0, 1.0]]), ("A",), np.array([0], dtype=np.int16), np.zeros(1),
                    np.array([NEUTRAL_RGB], dtype=np.uint8))
    surfaces = {}
    if floor is not None:
        surfaces["floor"] = SurfaceSet(np.asarray(floor, dtype=float), np.array([len(floor)], dtype=np.int32),
                                       np.array([0], dtype=np.int16))
    return Bundle(
        storeys=tuple(storeys), levels=tuple(float(i) for i in range(len(storeys))), surfaces=surfaces,
        grid=grid, layers=(), rooms=tuple(RoomLabel(r, 0, 1.0, 1.0, 0.0, True) for r in rooms),
        cell_m=1.0, work_plane_m=1.0,
    )


def test_a_concave_polygon_is_triangulated_without_folding():
    # An L: a 4 m x 4 m square less its 2 m x 2 m north-east corner, 12 m2.
    ell = [[0, 0, 0], [4, 0, 0], [4, 2, 0], [2, 2, 0], [2, 4, 0], [0, 4, 0]]

    payload = page.payload(_hand_bundle(floor=ell))

    ox, oy, oz, scale = payload["quant"]
    tris = _part(payload, "k_floor_pos", "<i2").reshape(-1, 3, 3) * scale + np.array([ox, oy, oz])
    assert tris.shape == (4, 3, 3)
    areas = [0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0])[2]) for t in tris]
    assert sum(areas) == pytest.approx(12.0, abs=1e-6)
    normals = _part(payload, "k_floor_nrm", "i1").reshape(-1, 3)
    assert {tuple(abs(n) for n in row) for row in normals.tolist()} == {(0, 0, 127)}


def test_too_many_storeys_or_rooms_for_the_payload():
    with pytest.raises(DeliverError, match="the 3D view holds at most 127 storeys; this case has 128"):
        page.payload(_hand_bundle(storeys=[f"S{i}" for i in range(128)]))
    with pytest.raises(DeliverError, match="the 3D view holds at most 65535 rooms; this case has 65536"):
        page.payload(_hand_bundle(rooms=["A"] + [f"R{i}" for i in range(65535)]))


# -- size ------------------------------------------------------------------------------

def test_the_size_warning():
    assert page.size_warning(25_000_000) is None
    assert page.size_warning(1_000) is None
    assert page.size_warning(26_000_000) == (
        "atlas.html is 26.0 MB, above 25 MB; a browser may be slow to open it"
    )


def test_script_json_leaves_no_less_than_sign_in_the_script_element():
    text = page._script_json({"name": "a <!-- b </script> c < d"})

    assert "<" not in text
    assert json.loads(text) == {"name": "a <!-- b </script> c < d"}
