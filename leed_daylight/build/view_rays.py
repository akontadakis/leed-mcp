"""The panorama of view directions sent from every Quality Views point
(QV spec 6.3): azimuth 0 to 360 degrees, elevation +/- half the vertical
span, both at one angular step. The step must divide 360 and half the
span exactly, or the panorama is refused (a step of 4 deg would stop the
60 deg band at +/-28 deg). Each direction is weighted by the cosine
of its elevation, so the weights follow solid angle (the "weighted by
solid angle" median of ClimateStudio's view distance).
"""
from __future__ import annotations

import math

import numpy as np


class ViewRayError(ValueError):
    """The panorama cannot be built from these settings."""


def panorama(step_deg: float, vertical_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions (m, 3) and weights (m,) summing to 1, azimuth-major."""
    if not (math.isfinite(step_deg) and 0 < step_deg <= 90):
        raise ViewRayError(f"step_deg must be 0 < a <= 90, got {step_deg!r}")
    if not (math.isfinite(vertical_deg) and 0 < vertical_deg <= 180):
        raise ViewRayError(f"vertical_deg must be 0 < v <= 180, got {vertical_deg!r}")
    n_azimuth = int(round(360.0 / step_deg))
    if abs(n_azimuth * step_deg - 360.0) > 1e-9:
        raise ViewRayError(
            f"views.angular_step_deg {step_deg!r} deg must divide 360 exactly, so the panorama closes"
        )
    n_half = int(round(vertical_deg / 2.0 / step_deg))
    if abs(n_half * step_deg - vertical_deg / 2.0) > 1e-9:
        raise ViewRayError(
            f"views.angular_step_deg {step_deg!r} deg must divide half the vertical span "
            f"({vertical_deg / 2.0:g} deg) exactly, so the panorama reaches +/-{vertical_deg / 2.0:g} deg"
        )
    azimuth = np.radians(step_deg * np.arange(n_azimuth))
    elevation = np.radians(step_deg * np.arange(-n_half, n_half + 1))
    az, el = np.meshgrid(azimuth, elevation, indexing="ij")
    az = az.ravel()
    el = el.ravel()
    directions = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    weights = np.cos(el)
    return directions, weights / weights.sum()


def rays_for_points(points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every direction from every point: all rays of point 0 first."""
    points = np.asarray(points, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ViewRayError(f"points must have shape (p, 3), got {points.shape}")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ViewRayError(f"directions must have shape (m, 3), got {directions.shape}")
    m = directions.shape[0]
    return np.repeat(points, m, axis=0), np.tile(directions, (points.shape[0], 1))
