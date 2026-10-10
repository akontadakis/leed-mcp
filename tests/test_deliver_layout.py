# tests/test_deliver_layout.py
"""The reports on split and merged spaces (merge-split spec 7; atlas spec 8
for the shared plan code): dashed lines in all three painters, split lines
on a plan, the storeys listing space rows only with every piece outlined
and one label per space, the layout sentences, and the Word Method section.

The projects are those of ``tests/merge_split_folder.py`` (run with the fake
engines) and hand-built rooms; every number is invented and round.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

pytest.importorskip("PIL", reason="the report extras are not installed: pip install -e '.[report]'")
from PIL import Image
from shapely.geometry import Point

import leed_daylight.drivers.option2 as option2_module
import leed_daylight.drivers.run as run_module
import leed_daylight.drivers.views as views_module
from deliver_folder import TWO_STOREY_CREDIT_RULES, Option1Fake, Option2Fake, build_deliver_project, doc_text
from leed_daylight.deliver.data import _storeys, layout_note, load_case
from leed_daylight.deliver.figures.base import get_renderer
from leed_daylight.deliver.figures.spec import (
    ROOM_RGB, SEGMENT_RGB, WHITE, Canvas, FigureError, Line, PlanSpec, Poly, Text, plan_canvas,
)
from leed_daylight.deliver.figures.svgtext import to_svg
from leed_daylight.deliver.word import common
from leed_daylight.project.layout import LaidOutSpace, apply_splits, lay_out
from leed_daylight.project.spaces import Space
from leed_daylight.project.splits import SplitLine
from leed_daylight.scene.primitives import Primitive
from leed_daylight.scene.storeys import Room, detect_rooms
from merge_split_folder import (
    CASE, MERGE_SCENE_RAD, MERGE_SPACES_CSV, build_merge_project, build_split_project, prepared_model,
)
from option2_folder import build_option2_project
from test_views_command import FakeTracer

MERGED_NOTE = (
    "Room North is two detected rooms treated as one space; each keeps its own grid, and blinds "
    "respond to the whole space."
)
#: The same sentence for a report whose option uses no blinds (Option 2, Quality Views).
MERGED_NOTE_NO_BLINDS = "Room North is two detected rooms treated as one space; each keeps its own grid."


@pytest.fixture(params=["matplotlib", "cairosvg", "pillow"])
def renderer(request):
    try:
        return get_renderer(request.param)
    except FigureError as exc:
        pytest.skip(f"{request.param} renderer not available here: {exc}")


def _image(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def _close(pixel, colour, tolerance=8) -> bool:
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(pixel, colour))


# -- dashed lines ------------------------------------------------------------


def test_a_dashed_line_is_painted_as_dashes(renderer):
    """10 px dashes and 10 px gaps from x = 20: x 20-30 drawn, 30-40 blank."""
    canvas = Canvas(200, 100, (Line(20.0, 50.0, 180.0, 50.0, SEGMENT_RGB, 3.0, dash=(10.0, 10.0)),))

    image = _image(renderer.paint(canvas))

    for x in (25, 45, 65, 165):
        assert _close(image.getpixel((x, 50)), SEGMENT_RGB), x
    for x in (35, 55, 75, 175):
        assert _close(image.getpixel((x, 50)), WHITE), x


def test_a_line_with_no_dash_is_solid(renderer):
    line = Line(20.0, 50.0, 180.0, 50.0, SEGMENT_RGB, 3.0)

    image = _image(renderer.paint(Canvas(200, 100, (line,))))

    assert line.dash == ()
    for x in (25, 35, 55, 175):
        assert _close(image.getpixel((x, 50)), SEGMENT_RGB), x


def test_svg_text_writes_a_dash_array_only_for_a_dashed_line():
    canvas = Canvas(200, 100, (
        Line(20.0, 50.0, 180.0, 50.0, SEGMENT_RGB, 3.0, dash=(10.0, 10.0)),
        Line(20.0, 70.0, 180.0, 70.0, SEGMENT_RGB, 3.0),
    ))

    text = to_svg(canvas)

    assert text.count('stroke-dasharray="10 10"') == 1
    assert text.count("stroke-dasharray") == 1


# -- split lines on a plan ---------------------------------------------------

OUTLINE = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]])


def _plan(**kwargs) -> PlanSpec:
    return PlanSpec(title="Plan", outlines=(OUTLINE,), labels=(("Office", 3.0, 2.0),),
                    points_xy=np.zeros((0, 2)), cell_m=0.6, north_rotation_deg=0.0, width_px=800, **kwargs)


def test_split_lines_are_drawn_dashed_right_after_the_room_outlines():
    bent = np.array([[3.0, 0.0], [3.0, 2.0], [4.0, 4.0]])

    canvas, layout = plan_canvas(_plan(split_lines=(bent,)))

    items = list(canvas.items)
    outline = max(i for i, item in enumerate(items) if isinstance(item, Poly) and item.stroke == ROOM_RGB)
    lines = items[outline + 1:outline + 3]
    assert lines == [
        Line(*layout.to_px(3.0, 0.0), *layout.to_px(3.0, 2.0), ROOM_RGB, 1.5, dash=(6.0, 4.0)),
        Line(*layout.to_px(3.0, 2.0), *layout.to_px(4.0, 4.0), ROOM_RGB, 1.5, dash=(6.0, 4.0)),
    ]
    assert isinstance(items[outline + 3], Text)


def test_a_plan_with_no_split_line_has_no_dashed_line():
    canvas, _ = plan_canvas(_plan())

    assert not [item for item in canvas.items if isinstance(item, Line) and item.dash]
    assert "stroke-dasharray" not in to_svg(canvas)


# -- the storeys: space rows, every piece, one label -------------------------


def _room(room_id, corners, area):
    return Room(room_id=room_id, storey=0, level_z=0.0, outline=np.array(corners, dtype=float),
                area_m2=area, ceiling_height_m=3.0)


def _space(room, cx, cy, merge_into=None):
    return Space(
        floor="L0", room=room, space_type="office", regularly_occupied=True, area_m2=10.0,
        centroid_x=cx, centroid_y=cy, level_z=0.0, view_preserving_shades=False,
        merge_into=merge_into, include={"base_case": True}, reason="",
    )


SMALL = _room("S0_R0", [(0, 0), (2, 0), (2, 2), (0, 2)], 4.0)
LARGE = _room("S0_R1", [(4, 0), (10, 0), (10, 4), (4, 4)], 24.0)
#: An L: the 6 x 6 m square at x 4-10, y 10-16 less its 4 x 4 m north-east corner (20 m2).
L_SHAPED = _room("S0_R2", [(4, 10), (10, 10), (10, 12), (6, 12), (6, 16), (4, 16)], 20.0)


def test_the_storeys_list_space_rows_only_and_outline_every_piece():
    spaces = [_space("Office", 1.0, 1.0), _space("Office Annex", 7.0, 2.0, merge_into="Office")]
    layout = lay_out(spaces, {"Office": SMALL, "Office Annex": LARGE})

    (storey,) = _storeys(spaces, layout)

    assert storey.rooms == ("Office",)
    assert [ring.tolist() for ring in storey.outlines["Office"]] == [SMALL.outline.tolist(), LARGE.outline.tolist()]
    assert dict(storey.split_lines) == {}


def test_a_merged_space_is_labelled_on_its_largest_piece():
    spaces = [_space("Office", 1.0, 1.0), _space("Office Annex", 7.0, 2.0, merge_into="Office")]

    (storey,) = _storeys(spaces, lay_out(spaces, {"Office": SMALL, "Office Annex": LARGE}))

    assert storey.centroids["Office"] == (7.0, 2.0)


def test_the_label_moves_inside_when_the_largest_pieces_centroid_is_outside_it():
    spaces = [_space("Office", 1.0, 1.0), _space("Office Annex", 5.0, 12.0, merge_into="Office")]

    (storey,) = _storeys(spaces, lay_out(spaces, {"Office": SMALL, "Office Annex": L_SHAPED}))

    # The L's centroid (6.2, 12.2) lies in its missing corner.
    assert not L_SHAPED.footprint.covers(Point(6.2, 12.2))
    assert L_SHAPED.footprint.contains(Point(*storey.centroids["Office"]))


def test_a_one_piece_space_keeps_its_spaces_csv_centroid():
    spaces = [_space("Office", 7.5, 2.5)]

    (storey,) = _storeys(spaces, lay_out(spaces, {"Office": LARGE}))

    assert storey.centroids["Office"] == (7.5, 2.5)


def _halves():
    """A 10 x 4 m floor cut along x = 5 m by splits.csv line 2 (run north:
    half a is the west), then its west half along y = 2 m by line 3 (run
    west: half aa is the south)."""
    corners = [(0, 0, 0), (10, 0, 0), (10, 4, 0), (0, 4, 0)]
    prims = [Primitive("slab", "polygon", "floor_a", (), (), tuple(float(v) for c in corners for v in c), "test")]
    roles = {"slab": "floor"}
    lines = (
        SplitLine(row=2, level_z=0.0, points=np.array([[5.0, -1.0], [5.0, 5.0]]), note=""),
        SplitLine(row=3, level_z=0.0, points=np.array([[5.0, 2.0], [-1.0, 2.0]]), note=""),
    )
    return apply_splits(detect_rooms(prims, roles), lines, prims, roles).rooms


def _halves_of_two_rooms():
    """Two 10 x 4 m floors 10 m apart, each cut in two across its length:
    the west floor along x = 5 m by splits.csv line 2, the east floor along
    x = 25 m by line 3 (both run north: half a is the west)."""
    prims, roles = [], {}
    for name, x0 in (("slab_w", 0.0), ("slab_e", 20.0)):
        corners = [(x0, 0, 0), (x0 + 10, 0, 0), (x0 + 10, 4, 0), (x0, 4, 0)]
        prims.append(Primitive(name, "polygon", f"floor_{name}", (), (), tuple(float(v) for c in corners for v in c),
                               "test"))
        roles[name] = "floor"
    lines = (
        SplitLine(row=2, level_z=0.0, points=np.array([[5.0, -1.0], [5.0, 5.0]]), note=""),
        SplitLine(row=3, level_z=0.0, points=np.array([[25.0, -1.0], [25.0, 5.0]]), note=""),
    )
    return apply_splits(detect_rooms(prims, roles), lines, prims, roles).rooms


def test_split_lines_are_the_drawn_lines_of_every_cut():
    aa, ab, b = _halves()
    assert [r.room_id for r in (aa, ab, b)] == ["S0_R0aa", "S0_R0ab", "S0_R0b"]
    spaces = [_space("South West", 2.5, 1.0), _space("North West", 2.5, 3.0), _space("East", 7.5, 2.0)]

    (storey,) = _storeys(spaces, lay_out(spaces, {"South West": aa, "North West": ab, "East": b}))

    assert [line.tolist() for line in storey.split_lines["South West"]] == [
        [[5.0, 0.0], [5.0, 4.0]], [[5.0, 2.0], [0.0, 2.0]],
    ]
    assert [line.tolist() for line in storey.split_lines["East"]] == [[[5.0, 0.0], [5.0, 4.0]]]
    # The line two halves share is one array, drawn once on a plan.
    assert storey.split_lines["East"][0] is storey.split_lines["South West"][0]
    assert storey.split_lines["North West"][1] is storey.split_lines["South West"][1]


# -- the layout sentences ----------------------------------------------------


def _laid(name, pieces, origin):
    return LaidOutSpace(name=name, storey=0, level_z=0.0, pieces=tuple(pieces), area_m2=0.0, origin=origin,
                        grid_source=(None,) * len(pieces), rows=(name,) * len(pieces))


def test_layout_note_wording_for_each_origin():
    aa, ab, b = _halves()

    assert layout_note(_laid("Office", [SMALL], "detected")) is None
    assert layout_note(_laid("Office 1", [SMALL, LARGE], "merged")) == (
        "Office 1 is two detected rooms treated as one space; each keeps its own grid, and blinds respond "
        "to the whole space."
    )
    assert layout_note(_laid("Open Office East", [b], "split")) == (
        "Open Office East was split from a detected room by line 2 of splits.csv."
    )
    assert layout_note(_laid("Open Office West", [aa], "split")) == (
        "Open Office West was split from a detected room by lines 2 and 3 of splits.csv."
    )
    assert layout_note(_laid("Office 2", [LARGE, b, SMALL], "split+merged")) == (
        "Office 2 is three rooms treated as one space, one or more of them split from a detected room by "
        "line 2 of splits.csv; each keeps its own grid, and blinds respond to the whole space."
    )


def test_a_space_of_halves_from_two_detected_rooms_says_detected_rooms():
    """Review M5: halves split from two different detected rooms are not
    "split from a detected room"; the sentence names the rooms in the plural."""
    halves = _halves_of_two_rooms()
    assert len(halves) == 4
    east_of_west_floor = next(h for h in halves if h.cuts[0].row == 2 and h.outline[:, 0].min() >= 5.0 - 1e-9)
    west_of_east_floor = next(h for h in halves if h.cuts[0].row == 3 and h.outline[:, 0].max() <= 25.0 + 1e-9)

    assert layout_note(_laid("Office 3", [east_of_west_floor, west_of_east_floor], "split+merged")) == (
        "Office 3 is two rooms treated as one space, one or more of them split from detected rooms by "
        "lines 2 and 3 of splits.csv; each keeps its own grid, and blinds respond to the whole space."
    )
    assert layout_note(_laid("Office 3", [east_of_west_floor, west_of_east_floor], "split+merged"),
                       grid=False) == (
        "Office 3 is two rooms treated as one space, one or more of them split from detected rooms by "
        "lines 2 and 3 of splits.csv."
    )


def test_without_blinds_the_note_drops_the_blind_clause_only():
    aa, ab, b = _halves()

    assert layout_note(_laid("Office 1", [SMALL, LARGE], "merged"), blinds=False) == (
        "Office 1 is two detected rooms treated as one space; each keeps its own grid."
    )
    assert layout_note(_laid("Office 2", [LARGE, b, SMALL], "split+merged"), blinds=False) == (
        "Office 2 is three rooms treated as one space, one or more of them split from a detected room by "
        "line 2 of splits.csv; each keeps its own grid."
    )
    assert layout_note(_laid("Open Office East", [b], "split"), blinds=False) == (
        layout_note(_laid("Open Office East", [b], "split"))
    )
    assert layout_note(_laid("Office", [SMALL], "detected"), blinds=False) is None


def test_grid_false_drops_the_grid_and_blind_clause_but_not_the_split_wording():
    """Controller ruling, pre-flight F2b: a space with no grid in this case
    (excluded from it) states neither the grid clause nor the blind clause;
    a split space's sentence, which states neither already, is unaffected."""
    aa, ab, b = _halves()

    assert layout_note(_laid("Office 1", [SMALL, LARGE], "merged"), grid=False) == (
        "Office 1 is two detected rooms treated as one space."
    )
    assert layout_note(_laid("Office 2", [LARGE, b, SMALL], "split+merged"), grid=False) == (
        "Office 2 is three rooms treated as one space, one or more of them split from a detected room by "
        "line 2 of splits.csv."
    )
    assert layout_note(_laid("Open Office East", [b], "split"), grid=False) == (
        layout_note(_laid("Open Office East", [b], "split"))
    )
    assert layout_note(_laid("Office", [SMALL], "detected"), grid=False) is None


def test_a_count_above_nine_is_written_in_digits():
    assert layout_note(_laid("Hall", [SMALL] * 10, "merged")).startswith("Hall is 10 detected rooms")
    assert layout_note(_laid("Hall", [SMALL] * 9, "merged")).startswith("Hall is nine detected rooms")


# -- the reports -------------------------------------------------------------


@pytest.fixture
def merged_case(tmp_path, write_example, write_wea):
    """The merge project with all three options run on fake engines."""
    folder = build_merge_project(tmp_path, write_example, write_wea)
    (folder / "credit_rules.yaml").write_text(TWO_STOREY_CREDIT_RULES)  # with a views section
    prepared = prepared_model(folder)
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared), out=io.StringIO())
    option2_module.option2_folder(folder, CASE, engine=Option2Fake(prepared), out=io.StringIO())
    views_module.run_views(folder, CASE, tracer=FakeTracer(), out=io.StringIO())
    return load_case(folder, CASE)


def test_plans_draw_every_piece_and_one_label(merged_case):
    (storey,) = merged_case.storeys
    option = merged_case.options["option1"]

    spec = common.storey_plan(merged_case, storey, option, title="Storey L0")
    canvas, _ = plan_canvas(spec)

    assert storey.rooms == ("Room North",)
    assert len(spec.outlines) == 2
    assert spec.labels == (("Room North", 3.0, 2.0),)
    assert [item.text for item in canvas.items if isinstance(item, Text)].count("Room North") == 1
    assert spec.points_xy.shape[0] == json.loads(option.results_path.read_text())["spaces"][0]["points"]


def test_a_split_plan_draws_the_shared_line_once(tmp_path, write_example, write_wea):
    folder = build_split_project(tmp_path, write_example, write_wea)
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared_model(folder)), out=io.StringIO())
    case = load_case(folder, CASE, options=("option1",))
    (storey,) = case.storeys

    spec = common.storey_plan(case, storey, case.options["option1"], title="Storey L0")

    assert storey.rooms == ("Room West", "Room Strip")
    assert [line.tolist() for line in spec.split_lines] == [[[4.5, 0.0], [4.5, 4.0]]]
    assert dict(case.layout_notes) == {
        "Room West": "Room West was split from a detected room by line 2 of splits.csv.",
        "Room Strip": "Room Strip was split from a detected room by line 2 of splits.csv.",
    }


@pytest.mark.parametrize("option, note", [
    ("option1", MERGED_NOTE), ("option2", MERGED_NOTE_NO_BLINDS), ("views", MERGED_NOTE_NO_BLINDS),
])
def test_the_method_section_lists_the_sentence_for_a_merged_space(merged_case, option, note):
    pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
    from leed_daylight.deliver.word import option1, option2, views

    builder = {"option1": option1, "option2": option2, "views": views}[option]
    built = builder.build_project(merged_case, get_renderer("pillow"))

    paragraphs = [p.text for p in built.document.paragraphs]
    method = paragraphs.index("Method")
    assert note in paragraphs[method:]
    assert dict(merged_case.layout_notes) == {"Room North": MERGED_NOTE}
    assert dict(merged_case.layout_notes_no_blinds) == {"Room North": MERGED_NOTE_NO_BLINDS}


@pytest.mark.parametrize("option", ["option2", "views"])
def test_the_option2_and_views_reports_say_nothing_of_blinds_responding(merged_case, option):
    pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
    from leed_daylight.deliver.word import option2, views

    builder = {"option2": option2, "views": views}[option]
    built = builder.build_project(merged_case, get_renderer("pillow"))

    assert "blinds respond" not in doc_text(built.document)


def test_the_method_section_has_no_sentence_for_a_detected_space(tmp_path, write_example, write_wea):
    pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
    from leed_daylight.deliver.word import option1

    case = load_case(build_deliver_project(tmp_path, write_example, write_wea, options=("option1",)))

    built = option1.build_project(case, get_renderer("pillow"))

    assert dict(case.layout_notes) == {}
    assert "treated as one space" not in doc_text(built.document)
    assert "splits.csv" not in doc_text(built.document)


# -- controller ruling (pre-flight F2): the blind clause needs an actual blind ----
#
# The brief's ``layout_note(space, blinds=True)`` always adds the blind
# clause for a merged or split+merged space in the Option 1 report. A
# controller ruling goes one step further: Option 1 adds the clause only
# when at least one of the space's window groups can carry a blind (a
# ``diffusing`` group never has one; a space with no window group has
# none). Otherwise Option 1 uses the same sentence as Option 2 and Quality
# Views. This project reuses the merge project's rooms and geometry
# (``merge_split_folder.MERGE_SCENE_RAD``/``MERGE_SPACES_CSV``) but with
# every glazing pane -- the south window of each room and the gap pane --
# built as a ``trans`` primitive under ``glazing_diffusing``, so every
# window group of the merged space "Room North" is diffusing glass (a
# ``glazing_diffusing`` pane needs a Radiance ``trans`` primitive, never a
# ``glass`` one: ``build.prepare`` refuses the mismatch).
_DIFFUSING_SCENE_RAD = (
    MERGE_SCENE_RAD.replace(
        "void glass glass_clear\n0\n0\n3  0.6 0.6 0.6\n",
        "void glass glass_clear\n0\n0\n3  0.6 0.6 0.6\n\n"
        "void trans glass_diffuse\n0\n0\n7  0.8 0.8 0.8  0 0  0.5  0\n",
    )
    .replace("glass_clear polygon window_ext", "glass_diffuse polygon window_ext")
    .replace("glass_clear polygon window_east", "glass_diffuse polygon window_east")
    .replace("glass_clear polygon window_gap", "glass_diffuse polygon window_gap")
)
_DIFFUSING_MATERIALS_CSV = (
    "modifier,primitive_type,role,note\n"
    "wall_paint,plastic,wall,\n"
    "ceiling_tile,plastic,ceiling,\n"
    "floor_carpet,plastic,floor,\n"
    "glass_diffuse,trans,glazing_diffusing,\n"
)


@pytest.fixture
def merged_case_all_diffusing(tmp_path, write_example, write_wea):
    """The merge project's rooms ("Room North" and "Room East", merged into
    one space), but every pane diffusing glass, so "Room North" has window
    groups and none of them can carry a blind."""
    def mutate(d):
        d["views"] = {"eye_height_m": 1.2, "eye_height_source": "Fictional example: seated eye height"}

    folder = build_option2_project(
        tmp_path, write_example, write_wea, spaces_csv=MERGE_SPACES_CSV, mutate=mutate,
        scene_rad=_DIFFUSING_SCENE_RAD,
    )
    (folder / "materials.csv").write_text(_DIFFUSING_MATERIALS_CSV)
    prepared = prepared_model(folder)
    run_module.run_folder(folder, CASE, engine=Option1Fake(prepared), out=io.StringIO())
    return load_case(folder, CASE, options=("option1",))


def test_option1_has_no_blind_clause_when_every_group_is_diffusing(merged_case_all_diffusing):
    case = merged_case_all_diffusing

    assert {g.room for g in case.groups} == {"Room North"}
    assert {g.kind for g in case.groups} == {"diffusing"}
    assert dict(case.layout_notes) == {"Room North": MERGED_NOTE_NO_BLINDS}


def test_the_option1_report_has_no_blind_clause_when_every_group_is_diffusing(merged_case_all_diffusing):
    pytest.importorskip("docx", reason="the report extras are not installed: pip install -e '.[report]'")
    from leed_daylight.deliver.word import option1

    built = option1.build_project(merged_case_all_diffusing, get_renderer("pillow"))

    assert "blinds respond" not in doc_text(built.document)
    assert MERGED_NOTE_NO_BLINDS in doc_text(built.document)
