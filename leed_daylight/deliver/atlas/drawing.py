# leed_daylight/deliver/atlas/drawing.py
"""A drawing under a storey plan of the atlas (atlas spec 8.1).

A drawing is a PNG or JPEG image in the project folder, named in the
`atlas:` section of project.yaml with two reference points: two pixels of
the image and the two model points (metres) they show. The two points fix
a similarity -- one scale, a rotation, a translation; no shear, no stretch --
with pixel y running down and model y up (``similarity``).

Where each refusal lives: ``project/config.py`` refuses a malformed
section and ``project/checks.py`` what the numbers alone show (an opacity
outside 0 to 1, two drawings for one storey, two points closer than 10
pixels or 0.5 m), both when project.yaml loads. ``load_drawings`` refuses
the rest, each naming the entry: a storey that is not in spaces.csv, an
image path that is absolute, climbs out with ``..`` or is a symlink leading
out of the project folder, an image that is missing, and a file that is
not PNG or JPEG by its signature (never by its name: a PDF renamed to
``.png`` is refused as a PDF).
"""
from __future__ import annotations

import pathlib
from collections.abc import Sequence

from ...project.config import DrawingSpec
from ..data import CaseData, DeliverError
from ..figures.spec import Underlay

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_PDF_SIGNATURE = b"%PDF-"
#: JPEG start-of-frame markers: every SOFn but DHT (C4), JPG (C8) and DAC (CC).
_JPEG_FRAMES = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
#: JPEG markers with no length field: TEM, RST0 to RST7, SOI, EOI.
_JPEG_BARE = frozenset({0x01, *range(0xD0, 0xDA)})


class DrawingError(DeliverError):
    """A drawing cannot be placed under a storey plan."""


def _what(data: bytes) -> str:
    if not data:
        return "the file is empty"
    if data.startswith(_PDF_SIGNATURE):
        return "the file is a PDF"
    return f"the file starts with bytes {data[:8].hex(' ')}"


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the first frame header after the start marker."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise DrawingError("a JPEG file whose segments cannot be read; the file is damaged")
        marker = data[i + 1]
        if marker == 0xFF:  # a fill byte before the marker
            i += 1
            continue
        if marker in _JPEG_BARE:
            i += 2
            continue
        if marker in _JPEG_FRAMES and i + 9 <= len(data):
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2:
            raise DrawingError("a JPEG file whose segments cannot be read; the file is damaged")
        i += 2 + length
    raise DrawingError("a JPEG file with no frame header (SOF); the file is damaged")


def image_info(data: bytes) -> tuple[str, int, int]:
    """(mime, width, height) of a PNG or JPEG image, read from its signature
    and header; a ``DrawingError`` for anything else."""
    if data.startswith(_PNG_SIGNATURE):
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise DrawingError("a PNG file with no image header (IHDR); the file is damaged")
        mime = "image/png"
        width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    elif data.startswith(_JPEG_SIGNATURE):
        mime = "image/jpeg"
        width, height = _jpeg_size(data)
    else:
        raise DrawingError(f"not a PNG or JPEG image ({_what(data)}); save the drawing as PNG or JPG")
    if width <= 0 or height <= 0:
        raise DrawingError(f"a {mime} image of {width} x {height} pixels; the file is damaged")
    return mime, width, height


def similarity(pixel: Sequence[Sequence[float]],
               model: Sequence[Sequence[float]]) -> tuple[float, float, float, float, float, float]:
    """The similarity taking pixel ``pixel[k]`` to model point ``model[k]``
    for both k, as ``Underlay.model_from_pixel`` ``(a, b, c, d, e, f)``:
    ``x = a*u + c*v + e``, ``y = b*u + d*v + f``. Pixel v runs down and model
    y up, so the drawing is mirrored once on the way, never stretched."""
    (u0, v0), (u1, v1) = pixel
    (x0, y0), (x1, y1) = model
    w0, w1 = complex(u0, -v0), complex(u1, -v1)
    z0, z1 = complex(x0, y0), complex(x1, y1)
    if w1 == w0:
        raise DrawingError("the two pixels are one point")
    if z1 == z0:
        raise DrawingError("the two model points are one point")
    s = (z1 - z0) / (w1 - w0)
    t = z0 - s * w0
    return (s.real, s.imag, s.imag, -s.real, t.real, t.imag)


def _image_path(root: pathlib.Path, image: str, entry: str) -> pathlib.Path:
    """``image`` inside ``root``: relative, no ``..``, present, and not a
    symlink leading out."""
    if pathlib.PurePosixPath(image).is_absolute() or pathlib.PureWindowsPath(image).anchor:
        raise DrawingError(f"{entry}: the image must be a path relative to the project folder, got {image!r}")
    if ".." in pathlib.PurePosixPath(image.replace("\\", "/")).parts:
        raise DrawingError(f"{entry}: the image path must not contain '..', got {image!r}")
    path = root / image
    if not path.is_file():
        raise DrawingError(f"{entry}: image not found: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise DrawingError(
            f"{entry}: {image!r} leads out of the project folder (to {path.resolve()}); "
            "copy the drawing into the project folder"
        ) from None
    return path


def place(root: pathlib.Path, index: int, drawing: DrawingSpec, labels: Sequence[str]) -> Underlay:
    """Entry ``index`` of `atlas.drawings`, ``drawing``, as the ``Underlay``
    of its storey: its storey among ``labels`` (spaces.csv's storeys), its
    image inside ``root`` and a PNG or JPEG, its two points a similarity; a
    ``DrawingError`` naming the entry otherwise. The wizard's drawings step
    places an entry with it as the atlas does."""
    entry = f"atlas.drawings[{index}] (storey {drawing.storey!r}, image {drawing.image!r})"
    if drawing.storey not in labels:
        raise DrawingError(
            f"{entry}: storey {drawing.storey!r} is not in spaces.csv; its storeys are {', '.join(labels)}"
        )
    data = _image_path(pathlib.Path(root), drawing.image, entry).read_bytes()
    try:
        mime, width, height = image_info(data)
        matrix = similarity(drawing.pixel, drawing.model)
    except DrawingError as exc:
        raise DrawingError(f"{entry}: {exc}") from None
    return Underlay(data, mime, width, height, matrix, drawing.opacity)


def load_drawings(case: CaseData) -> dict[str, Underlay]:
    """Each `atlas.drawings` entry of the case's project, as the ``Underlay``
    of its storey, keyed by storey label; a ``DrawingError`` naming the
    entry for the first one that cannot be placed."""
    root = pathlib.Path(case.project.root)
    labels = [storey.label for storey in case.storeys]
    return {
        drawing.storey: place(root, i, drawing, labels)
        for i, drawing in enumerate(case.project.atlas.drawings)
    }
