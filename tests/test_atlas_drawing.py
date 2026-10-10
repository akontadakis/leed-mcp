# tests/test_atlas_drawing.py
"""The `atlas:` section and the drawing under a storey plan (atlas spec 8.1).

`project.yaml` shape is refused by `project/config.py`; what the numbers
alone show (opacity, one drawing per storey, the two points too close) by
`project/checks.py`, both as `ProjectError` from `load_project`. What needs
the files or the case (the storey, the image path, the image itself) is
refused by `deliver/atlas/drawing.py` as `DrawingError`. Every refusal names
the entry. Every image here is made in the test; every number is invented
and round.
"""
from __future__ import annotations

import datetime
import io
import struct
import types
import zlib

import pytest

from conftest import symlink_or_skip
from deliver_folder import build_deliver_project
from leed_daylight.constants import FILL_ME
from leed_daylight.deliver.atlas.drawing import DrawingError, image_info, load_drawings, similarity
from leed_daylight.deliver.data import CaseData, DeliverError, StoreyPlan, load_case
from leed_daylight.deliver.figures.spec import Underlay
from leed_daylight.project.config import AtlasSpec, DrawingSpec, ProjectError, load_project

#: Pixel (100, 900) is model (0, 0) and pixel (1100, 900) is model (20, 0):
#: 0.02 m per pixel, the drawing upright.
POINTS = [{"pixel": [100, 900], "model": [0.0, 0.0]}, {"pixel": [1100, 900], "model": [20.0, 0.0]}]


def png_bytes(width: int, height: int, rgb=(255, 255, 255)) -> bytes:
    """A real, decodable RGB PNG of one colour."""
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def jpeg_header(width: int, height: int) -> bytes:
    """The segments of a JPEG up to its frame header (all ``image_info`` reads)."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"


def _drawing(**changes) -> dict:
    entry = {"storey": "L0", "image": "drawings/level0.png", "points": [dict(p) for p in POINTS]}
    entry.update(changes)
    return entry


def _project_dir(tmp_path, write_example, *drawings, image=None):
    def mutate(d):
        d["atlas"] = {"drawings": list(drawings)}
        return d

    folder = write_example(tmp_path, mutate).parent
    (folder / "drawings").mkdir()
    (folder / "drawings" / "level0.png").write_bytes(png_bytes(40, 30) if image is None else image)
    return folder


def _load(tmp_path, write_example, *drawings, image=None):
    return load_project(_project_dir(tmp_path, write_example, *drawings, image=image) / "project.yaml")


def _refused(tmp_path, write_example, section) -> str:
    def mutate(d):
        d["atlas"] = section
        return d

    with pytest.raises(ProjectError) as excinfo:
        load_project(write_example(tmp_path, mutate))
    return str(excinfo.value)


def _case(project, labels=("L0", "L1")) -> CaseData:
    """A case with only what ``load_drawings`` reads: the project and the storeys."""
    empty = types.MappingProxyType({})
    storeys = tuple(
        StoreyPlan(label, 4.0 * i, (f"Room {label}",), empty, empty) for i, label in enumerate(labels)
    )
    return CaseData(
        folder=project.root, case="base_case", created=datetime.date(2026, 1, 1), project=project,
        rules=empty, spaces=(), storeys=storeys, groups=(), glazing=empty, options=empty, skipped=empty,
        warnings=(),
    )


# -- project.yaml: the section --------------------------------------------------


def test_without_an_atlas_section_there_is_no_drawing(example_project_dir):
    project = load_project(example_project_dir / "project.yaml")
    assert project.atlas == AtlasSpec()
    assert project.atlas.drawings == ()


def test_a_drawing_loads_with_the_default_opacity(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing())
    assert project.atlas.drawings == (
        DrawingSpec(storey="L0", image="drawings/level0.png", pixel=((100.0, 900.0), (1100.0, 900.0)),
                    model=((0.0, 0.0), (20.0, 0.0)), opacity=0.35),
    )


def test_a_drawing_takes_its_own_opacity(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing(opacity=0.5), _drawing(storey="L1", opacity=1))
    assert [d.opacity for d in project.atlas.drawings] == [0.5, 1.0]


@pytest.mark.parametrize("section, message", [
    (["L0"], "atlas must be a mapping"),
    ({"colour": "grey"}, "unknown key 'atlas.colour'"),
    ({"drawings": "level0.png"}, "atlas.drawings must be a list"),
    ({"drawings": ["level0.png"]}, "atlas.drawings[0] must be a mapping"),
    ({"drawings": [_drawing(scale=2)]}, "unknown key 'atlas.drawings[0].scale'"),
    ({"drawings": [{k: v for k, v in _drawing().items() if k != "storey"}]},
     "missing required key 'atlas.drawings[0].storey'"),
    ({"drawings": [_drawing(storey=1)]}, "atlas.drawings[0].storey must be a string"),
    ({"drawings": [{k: v for k, v in _drawing().items() if k != "image"}]},
     "missing required key 'atlas.drawings[0].image'"),
    ({"drawings": [_drawing(points=POINTS[:1])]}, "atlas.drawings[0].points must be a list of exactly two points"),
    ({"drawings": [_drawing(points=[[100, 900], [1100, 900]])]}, "atlas.drawings[0].points[0] must be a mapping"),
    ({"drawings": [_drawing(points=[POINTS[0], {"pixel": [1100, 900]}])]},
     "missing required key 'atlas.drawings[0].points[1].model'"),
    ({"drawings": [_drawing(points=[POINTS[0], {"pixel": [1100], "model": [20.0, 0.0]}])]},
     "atlas.drawings[0].points[1].pixel must be a list of two numbers"),
    ({"drawings": [_drawing(points=[POINTS[0], {"pixel": [1100, 900], "model": ["20", 0.0]}])]},
     "atlas.drawings[0].points[1].model[0] must be a number"),
    ({"drawings": [_drawing(opacity="light")]}, "atlas.drawings[0].opacity must be a number"),
])
def test_a_malformed_atlas_section_is_refused_by_name(tmp_path, write_example, section, message):
    assert message in _refused(tmp_path, write_example, section)


@pytest.mark.parametrize("opacity", [1.5, -0.1])
def test_an_opacity_outside_0_to_1_is_refused(tmp_path, write_example, opacity):
    message = _refused(tmp_path, write_example, {"drawings": [_drawing(opacity=opacity)]})
    assert f"atlas.drawings[0].opacity must be 0 <= o <= 1, got {opacity}" in message


def test_two_drawings_for_one_storey_are_refused(tmp_path, write_example):
    message = _refused(tmp_path, write_example, {"drawings": [_drawing(), _drawing()]})
    assert "atlas.drawings[1]: storey 'L0' already has a drawing (atlas.drawings[0])" in message


def test_two_pixels_closer_than_10_are_refused(tmp_path, write_example):
    points = [POINTS[0], {"pixel": [103, 904], "model": [20.0, 0.0]}]
    message = _refused(tmp_path, write_example, {"drawings": [_drawing(points=points)]})
    assert "atlas.drawings[0].points: the two pixels are 5 pixels apart; place them at least 10" in message


def test_two_model_points_closer_than_half_a_metre_are_refused(tmp_path, write_example):
    points = [POINTS[0], {"pixel": [1100, 900], "model": [0.3, 0.0]}]
    message = _refused(tmp_path, write_example, {"drawings": [_drawing(points=points)]})
    assert "atlas.drawings[0].points: the two model points are 0.3 m apart" in message


def test_fill_me_in_the_atlas_section_is_refused(tmp_path, write_example):
    message = _refused(tmp_path, write_example, {"drawings": [_drawing(image=FILL_ME)]})
    assert "atlas.drawings[0].image" in message


# -- the image: its file signature and size --------------------------------------


def test_a_png_is_read_from_its_header():
    assert image_info(png_bytes(40, 30)) == ("image/png", 40, 30)


def test_a_jpeg_is_read_from_its_frame_header():
    assert image_info(jpeg_header(640, 480)) == ("image/jpeg", 640, 480)


def test_a_jpeg_written_by_pillow_is_read():
    pil = pytest.importorskip("PIL.Image", reason="the report extras are not installed")
    buffer = io.BytesIO()
    pil.new("RGB", (64, 48), (200, 200, 200)).save(buffer, format="JPEG")
    assert image_info(buffer.getvalue()) == ("image/jpeg", 64, 48)


@pytest.mark.parametrize("data, message", [
    (PDF_BYTES, "not a PNG or JPEG image (the file is a PDF)"),
    (b"GIF89a" + b"\x00" * 16, "not a PNG or JPEG image (the file starts with bytes 47 49 46 38 39 61 00 00)"),
    (b"", "not a PNG or JPEG image (the file is empty)"),
    (png_bytes(40, 30)[:20], "a PNG file with no image header"),
    (b"\xff\xd8\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00\xff\xd9", "a JPEG file with no frame header"),
])
def test_anything_else_is_refused(data, message):
    with pytest.raises(DrawingError, match="^" + message.replace("(", r"\(").replace(")", r"\)")):
        image_info(data)


def test_a_drawing_error_is_a_deliver_error():
    assert issubclass(DrawingError, DeliverError)


# -- the placement: a similarity from two points ----------------------------------


def _apply(matrix, u, v):
    a, b, c, d, e, f = matrix
    return a * u + c * v + e, b * u + d * v + f


def test_the_two_reference_points_map_exactly():
    matrix = similarity(((100.0, 900.0), (1100.0, 900.0)), ((0.0, 0.0), (20.0, 0.0)))
    assert _apply(matrix, 100.0, 900.0) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert _apply(matrix, 1100.0, 900.0) == pytest.approx((20.0, 0.0), abs=1e-9)


def test_a_third_pixel_lands_within_a_millimetre():
    # 500 pixels up the drawing is 10 m up the model (pixel y runs down).
    matrix = similarity(((100.0, 900.0), (1100.0, 900.0)), ((0.0, 0.0), (20.0, 0.0)))
    assert _apply(matrix, 100.0, 400.0) == pytest.approx((0.0, 10.0), abs=0.001)


def test_a_rotated_drawing_is_turned_and_never_stretched():
    # The drawing's x axis runs along the model's +y: turned 90 degrees.
    matrix = similarity(((100.0, 900.0), (1100.0, 900.0)), ((0.0, 0.0), (0.0, 20.0)))
    assert _apply(matrix, 100.0, 400.0) == pytest.approx((-10.0, 0.0), abs=0.001)
    a, b, c, d, _, _ = matrix
    assert a * a + b * b == pytest.approx(c * c + d * d)  # one scale on both axes
    assert a * c + b * d == pytest.approx(0.0, abs=1e-12)  # no shear


def test_a_drawing_seen_in_a_mirror_is_impossible():
    # The determinant is negative: pixel y down turns into model y up.
    a, b, c, d, _, _ = similarity(((0.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (5.0, 0.0)))
    assert a * d - b * c < 0.0


def test_coincident_points_are_refused():
    with pytest.raises(DrawingError, match="the two pixels are one point"):
        similarity(((100.0, 900.0), (100.0, 900.0)), ((0.0, 0.0), (20.0, 0.0)))
    with pytest.raises(DrawingError, match="the two model points are one point"):
        similarity(((100.0, 900.0), (1100.0, 900.0)), ((5.0, 5.0), (5.0, 5.0)))


# -- load_drawings: the files and the case ------------------------------------


def test_a_drawing_becomes_the_underlay_of_its_storey(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing(opacity=0.5))

    drawings = load_drawings(_case(project))

    assert list(drawings) == ["L0"]
    underlay = drawings["L0"]
    assert isinstance(underlay, Underlay)
    assert (underlay.mime, underlay.width, underlay.height, underlay.opacity) == ("image/png", 40, 30, 0.5)
    assert underlay.data == png_bytes(40, 30)
    assert _apply(underlay.model_from_pixel, 1100.0, 900.0) == pytest.approx((20.0, 0.0), abs=1e-9)


def test_no_atlas_section_gives_no_underlay(example_project_dir):
    assert load_drawings(_case(load_project(example_project_dir / "project.yaml"))) == {}


def test_a_storey_not_in_spaces_csv_is_refused(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing(storey="L9"))
    with pytest.raises(DrawingError, match=r"atlas.drawings\[0\] \(storey 'L9'.*storey 'L9' is not in spaces.csv"):
        load_drawings(_case(project))


def test_a_missing_image_is_refused(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing(image="drawings/level9.png"))
    with pytest.raises(DrawingError, match=r"atlas.drawings\[0\] .*image not found: .*level9.png"):
        load_drawings(_case(project))


@pytest.mark.parametrize("image", ["/elsewhere/level0.png", "C:\\drawings\\level0.png", "\\drawings\\level0.png"])
def test_an_absolute_image_path_is_refused(tmp_path, write_example, image):
    project = _load(tmp_path, write_example, _drawing(image=image))
    with pytest.raises(DrawingError, match="must be a path relative to the project folder"):
        load_drawings(_case(project))


@pytest.mark.parametrize("image", ["../level0.png", "drawings/../../level0.png", "drawings\\..\\..\\level0.png"])
def test_an_image_path_that_climbs_out_is_refused(tmp_path, write_example, image):
    project = _load(tmp_path, write_example, _drawing(image=image))
    with pytest.raises(DrawingError, match=r"must not contain '\.\.'"):
        load_drawings(_case(project))


def test_a_symlink_leading_out_of_the_folder_is_refused(tmp_path, write_example):
    outside = tmp_path / "outside.png"
    outside.write_bytes(png_bytes(40, 30))
    folder = _project_dir(tmp_path, write_example, _drawing(image="drawings/linked.png"))
    symlink_or_skip(folder / "drawings" / "linked.png", outside)
    with pytest.raises(DrawingError, match=r"atlas.drawings\[0\] .*leads out of the project folder"):
        load_drawings(_case(load_project(folder / "project.yaml")))


def test_a_symlink_inside_the_folder_is_followed(tmp_path, write_example):
    folder = _project_dir(tmp_path, write_example, _drawing(image="drawings/linked.png"))
    symlink_or_skip(folder / "drawings" / "linked.png", folder / "drawings" / "level0.png")
    assert load_drawings(_case(load_project(folder / "project.yaml")))["L0"].width == 40


def test_a_renamed_pdf_is_refused_by_its_signature(tmp_path, write_example):
    """Review Focus 4: a drawing saved as PDF and renamed to .png is refused
    by what the file holds, not by its name, and the refusal names the entry."""
    project = _load(tmp_path, write_example, _drawing(), image=PDF_BYTES)
    with pytest.raises(DrawingError) as excinfo:
        load_drawings(_case(project))
    message = str(excinfo.value)
    assert "atlas.drawings[0] (storey 'L0', image 'drawings/level0.png')" in message
    assert "not a PNG or JPEG image (the file is a PDF)" in message
    assert "save the drawing as PNG or JPG" in message


def test_a_jpeg_named_png_is_accepted_as_a_jpeg(tmp_path, write_example):
    project = _load(tmp_path, write_example, _drawing(), image=jpeg_header(640, 480))
    assert load_drawings(_case(project))["L0"].mime == "image/jpeg"


def test_a_model_check_case_carries_its_drawing(tmp_path, write_example, write_wea):
    folder = build_deliver_project(tmp_path, write_example, write_wea, options=())
    (folder / "drawings").mkdir()
    (folder / "drawings" / "level0.png").write_bytes(png_bytes(40, 30))
    with open(folder / "project.yaml", "a") as handle:
        handle.write(
            "atlas:\n  drawings:\n    - storey: L0\n      image: drawings/level0.png\n      points:\n"
            "        - {pixel: [100, 900], model: [0.0, 0.0]}\n"
            "        - {pixel: [1100, 900], model: [20.0, 0.0]}\n"
        )

    drawings = load_drawings(load_case(folder, require_results=False))

    assert list(drawings) == ["L0"]
