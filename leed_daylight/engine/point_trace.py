# leed_daylight/engine/point_trace.py
"""Illuminance at grid points under one sky, for Option 2 (layer 4).

One ``oconv`` (materials, scene, sky) and one ``rtrace -I`` over every
point, each point facing up. The result is cached beside a fingerprint of
the files it came from, the points, the Radiance settings and the Radiance
version, so a stopped run resumes and a changed sky is traced again.
``processes`` is not in the fingerprint: it changes the speed, not the
method. Radiance RGB becomes lux here, through ``units.rgb_to_lux``.
"""
from __future__ import annotations

import pathlib
import shlex

import numpy as np

from .. import units
from .fingerprint import delete_sidecar, fingerprint_of, is_current, write_sidecar
from .radiance import Radiance, RadianceError, process_count


class PointTraceError(RadianceError):
    """A point trace's inputs or rtrace's answer are unusable."""


def _points_text(points: np.ndarray) -> bytes:
    lines = [f"{float(x)!r} {float(y)!r} {float(z)!r} 0 0 1" for x, y, z in points]
    return ("\n".join(lines) + "\n").encode("ascii")


def _parse_rgb(out: bytes, expected: int) -> np.ndarray:
    lines = [line for line in out.decode("ascii", errors="replace").splitlines() if line.strip()]
    if len(lines) != expected:
        raise PointTraceError(
            f"rtrace returned {len(lines)} line(s) for {expected} point(s)"
        )
    try:
        rgb = np.array([[float(v) for v in line.split()[:3]] for line in lines])
    except ValueError:
        raise PointTraceError("rtrace returned a line that is not three numbers") from None
    if rgb.shape != (expected, 3) or not np.all(np.isfinite(rgb)):
        raise PointTraceError("rtrace returned a line that is not three finite numbers")
    return rgb


def trace_points(
    rad: Radiance, *, materials: pathlib.Path, scene: pathlib.Path, sky: pathlib.Path,
    points: np.ndarray, ambient: str, direct: str, work_dir: pathlib.Path, name: str,
    processes: int = 1,
) -> np.ndarray:
    """Lux at every row of ``points`` ((n, 3), metres) under ``sky``."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PointTraceError(f"points must have shape (n, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise PointTraceError("no points to trace")
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    stdin = _points_text(points)
    points_file = work_dir / f"{name}.pts"
    points_file.write_bytes(stdin)
    result = work_dir / f"{name}.lux.npy"
    fp = fingerprint_of(
        {"materials": pathlib.Path(materials), "scene": pathlib.Path(scene),
         "sky": pathlib.Path(sky), "points": points_file},
        {"kind": "option2_point_lux", "ambient": ambient, "direct": direct,
         "radiance_version": rad.version},
    )
    if is_current(result, fp):
        return np.load(result)

    delete_sidecar(result)
    octree = work_dir / f"{name}.oct"
    rad.run(["oconv", str(materials), str(scene), str(sky)], stdout_path=octree)
    out = rad.run(
        ["rtrace", "-n", str(process_count(processes)), "-h", "-I+", "-ov",
         *shlex.split(ambient), *shlex.split(direct), str(octree)],
        stdin=stdin,
    )
    lux = units.rgb_to_lux(_parse_rgb(out, points.shape[0]))
    np.save(result, lux)
    write_sidecar(result, fp)
    return lux
