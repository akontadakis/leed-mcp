# tests/golden_figures.py
"""Golden Word-report figures: the atlas work must leave every one byte for byte the same.

The invented project of ``tests/deliver_folder.py`` (all three options) is
built, and every figure its six Word reports make (the project report and
the storey report of Option 1, Option 2 and Quality Views) is recorded as
the display list (``Canvas``) the builder hands its painter. Each canvas is
then painted three ways: the matplotlib painter, the Pillow painter, and
the pure SVG writer ``svgtext.to_svg``. Those outputs were written once, on
master before the atlas work, into ``tests/fixtures/golden_figures/``:

- ``sha256.json``: the sha256 of every output (every figure, three ways);
- the bytes of the figures in ``KEPT``: one storey plan per option, the
  model-grid and window-group plans, and one bar chart. Kept small on
  purpose; the digest list covers the rest;
- ``pixels_sha256.json``: the sha256 of the decoded pixels (mode, size and
  raw bytes, ``pixel_digest``) of every PNG. Written later than the rest
  (``--write-pixels``), on macOS, from a tree whose every PNG still
  matched ``sha256.json`` byte for byte, so it holds master's pixels. It
  lets a system whose PNG encoder compresses the same pixels into other
  bytes (Linux, see ``tests/test_golden_figures.py``) still check every
  figure.

``tests/test_golden_figures.py`` builds the same figures on the code under
test and compares them. Nothing in them depends on the folder they are
built in. They do depend on the installed matplotlib and Pillow (fonts,
PNG encoder): after an upgrade of either, find what changed; never
regenerate the goldens on new code.

Generator mode (run once, on the commit the goldens are taken from; it
refuses to overwrite an existing golden folder)::

    PYTHONPATH=.:tests .venv/bin/python tests/golden_figures.py --write

and, once, for the pixel digests (refuses to overwrite them, and refuses
unless every PNG still matches its recorded byte digest)::

    PYTHONPATH=.:tests .venv/bin/python tests/golden_figures.py --write-pixels
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import pathlib
import sys
import tempfile

from conftest import write_example_project, write_wea_file
from deliver_folder import build_deliver_project
from leed_daylight.deliver.data import load_case
from leed_daylight.deliver.figures.base import get_renderer
from leed_daylight.deliver.figures.svgtext import to_svg
from leed_daylight.deliver.word import option1, option2, views

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "golden_figures"
DIGESTS = "sha256.json"
PIXEL_DIGESTS = "pixels_sha256.json"
STOREY = "L0"
TODAY = datetime.date(2026, 1, 1)
BUILDERS = {"option1": option1, "option2": option2, "views": views}
#: The three ways each canvas is written: file suffix -> painter.
SUFFIXES = (".matplotlib.png", ".pillow.png", ".svg")

#: Figures kept as bytes, by "<option>/<report>/<figure name without .png>".
KEPT = (
    "option1/project/storeys_sda",
    "option1/project/L0_grid",
    "option1/project/L0_zones",
    "option1/project/L0_sda",
    "option2/project/L0_sky_03-28_0900",
    "views/project/L0_quality",
)


class _Recorder:
    """A painter that records every canvas and paints it with Pillow (the
    Word builders read the PNG's size back)."""

    name = "pillow"

    def __init__(self):
        self._pillow = get_renderer("pillow")
        self.canvases = []

    def paint(self, canvas) -> bytes:
        self.canvases.append(canvas)
        return self._pillow.paint(canvas)


def figure_canvases(folder) -> dict[str, object]:
    """Every figure of the six Word reports, by "<option>/<report>/<name>"."""
    case = load_case(folder, today=TODAY)
    canvases = {}
    for option, builder in BUILDERS.items():
        for report in ("project", "storey"):
            recorder = _Recorder()
            if report == "project":
                built = builder.build_project(case, recorder)
            else:
                built = builder.build_storey(case, STOREY, recorder)
            if len(built.figures) != len(recorder.canvases):
                raise AssertionError(f"{option} {report}: a figure was not painted through the recorder")
            for (name, _), canvas in zip(built.figures, recorder.canvases):
                canvases[f"{option}/{report}/{name.removesuffix('.png')}"] = canvas
    return canvases


def render_figures(tmp_path) -> dict[str, bytes]:
    """Build the invented project under ``tmp_path`` and write every figure
    three ways: output name -> bytes."""
    folder = build_deliver_project(pathlib.Path(tmp_path), write_example_project, write_wea_file)
    mpl, pil = get_renderer("matplotlib"), get_renderer("pillow")
    out = {}
    for key, canvas in figure_canvases(folder).items():
        out[f"{key}.matplotlib.png"] = mpl.paint(canvas)
        out[f"{key}.pillow.png"] = pil.paint(canvas)
        out[f"{key}.svg"] = to_svg(canvas).encode("utf-8")
    return out


def kept_names() -> tuple[str, ...]:
    return tuple(f"{key}{suffix}" for key in KEPT for suffix in SUFFIXES)


def digests(rendered: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest() for name, data in sorted(rendered.items())}


def decode_png(data: bytes):
    """The PNG ``data`` decoded by Pillow, loaded."""
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def pixel_digest(data: bytes) -> str:
    """The sha256 of the PNG ``data``'s decoded pixels: its mode, its size
    and its raw pixel bytes, nothing of how they were compressed."""
    image = decode_png(data)
    head = f"{image.mode} {image.size[0]}x{image.size[1]}\n".encode("ascii")
    return hashlib.sha256(head + image.tobytes()).hexdigest()


def pixel_digests(rendered: dict[str, bytes]) -> dict[str, str]:
    return {name: pixel_digest(data) for name, data in sorted(rendered.items()) if name.endswith(".png")}


def write_pixel_digests(golden_dir=GOLDEN_DIR) -> None:
    """Generator mode for ``PIXEL_DIGESTS``: refuses if the file exists, or
    if any figure no longer matches its recorded byte digest (its pixels
    would then not be master's)."""
    golden_dir = pathlib.Path(golden_dir)
    target = golden_dir / PIXEL_DIGESTS
    if target.exists():
        raise SystemExit(f"{target} already exists; it is written once, never refreshed")
    with tempfile.TemporaryDirectory() as tmp:
        rendered = render_figures(tmp)
    recorded = json.loads((golden_dir / DIGESTS).read_text(encoding="utf-8"))
    changed = [name for name, digest in digests(rendered).items() if recorded.get(name) != digest]
    if changed:
        raise SystemExit(f"these figures differ from master, so their pixels are not master's: {changed}")
    target.write_text(json.dumps(pixel_digests(rendered), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {len(pixel_digests(rendered))} pixel digests to {target}")


def write_goldens(golden_dir=GOLDEN_DIR) -> None:
    """Generator mode: refuses, before writing anything, if ``golden_dir`` exists."""
    golden_dir = pathlib.Path(golden_dir)
    if golden_dir.exists():
        raise SystemExit(
            f"{golden_dir} already exists; goldens are written once, on the commit "
            "they are taken from, never refreshed"
        )
    with tempfile.TemporaryDirectory() as tmp:
        rendered = render_figures(tmp)
    for name in kept_names():
        path = golden_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered[name])
    (golden_dir / DIGESTS).write_text(json.dumps(digests(rendered), indent=2, sort_keys=True) + "\n", newline="\n")
    print(f"wrote {len(kept_names())} files of {len(KEPT)} kept figures and {len(rendered)} digests to {golden_dir}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write_goldens()
    elif sys.argv[1:] == ["--write-pixels"]:
        write_pixel_digests()
    else:
        raise SystemExit("usage: PYTHONPATH=.:tests .venv/bin/python tests/golden_figures.py --write | --write-pixels")
