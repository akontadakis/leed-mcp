# leed_daylight/deliver/figures/base.py
"""The renderer interface (reports spec R3): one painter per library.

A painter turns a ``Canvas`` into PNG bytes of exactly ``canvas.width`` x
``canvas.height`` pixels. The library is imported only when its painter is
asked for, so a missing library is a named refusal, never an import error at
start-up.

Pre-flight ruling F12: a painter module's own import failure is converted to
a friendly ``FigureError`` only when it is the *expected* library that is
missing (matched by the failing import's top-level module name, e.g.
``matplotlib``, ``cairosvg``/``cairocffi``, ``PIL``). Any other ``ImportError``
-- a real bug in the painter module, a typo, a stray import -- re-raises
unchanged, so it fails loudly instead of looking like a missing optional
dependency.
"""
from __future__ import annotations

import importlib
from typing import Protocol

from ...project.config import RENDERERS
from .spec import BarSpec, Canvas, FigureError, PlanSpec, bar_canvas, plan_canvas

#: renderer name -> (module inside this package, class name, top-level
#: module names that count as "the expected library" for that renderer).
_PAINTERS = {
    "matplotlib": ("mpl", "MatplotlibRenderer", ("matplotlib",)),
    "cairosvg": ("svg", "CairoSvgRenderer", ("cairosvg", "cairocffi")),
    "pillow": ("pil", "PillowRenderer", ("PIL",)),
}


class Renderer(Protocol):
    name: str

    def paint(self, canvas: Canvas) -> bytes: ...


def get_renderer(name: str) -> Renderer:
    if name not in _PAINTERS:
        raise FigureError(f"unknown renderer {name!r}; expected one of {', '.join(RENDERERS)}")
    module_name, class_name, libraries = _PAINTERS[name]
    try:
        module = importlib.import_module(f"{__package__}.{module_name}")
    except ImportError as exc:
        top = (exc.name or "").split(".")[0]
        if top not in libraries:
            raise
        raise FigureError(
            f"the {name} renderer needs the Python package {top!r} ({exc}); "
            'install the report extras: pip install -e ".[report]"'
        ) from None
    except OSError as exc:
        # cairosvg imports, then fails to load the system cairo library.
        raise FigureError(
            f"the {name} renderer needs the system cairo library ({exc}); "
            "install it (macOS: brew install cairo) or choose another renderer"
        ) from None
    return getattr(module, class_name)()


def render_plan(spec: PlanSpec, renderer: Renderer) -> bytes:
    canvas, _ = plan_canvas(spec)
    return renderer.paint(canvas)


def render_bar(spec: BarSpec, renderer: Renderer) -> bytes:
    return renderer.paint(bar_canvas(spec))
