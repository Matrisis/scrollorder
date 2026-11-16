"""Ordered curves of one slice: where each segment (a TIFXYZ surface) crosses the plane of the image.

A TIFXYZ surface is a grid of 3D points.  Cut at a height z0, each of its columns gives one point of
the curve, so the curve keeps the order of the grid and carries, point by point, the arc position
along the grid (``u_mm``) and the unit normal of the surface (``normal3d``).  Points the grid does not
give (no crossing, invalid points) are NaN rows, never dropped: the index of a point is its column.

Grid convention: z grows along the rows of the grid, the columns run along the curve.  A grid stored
the other way round is transposed first (see ``curve_at_z``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scrollorder.dataconf import DataConfig
from scrollorder.io_real import read_tifxyz
from scrollorder.thresholds import FOLD_FRAC
from scrollorder.volume import SliceImage, l0_to_level


@dataclass(eq=False)
class Curve:
    """One segment cut by the plane of a slice: ``xy`` (N, 2) in pixels of the slice's level (NaN in the
    holes), ``u_mm`` (N,) the position along the grid, ``normal3d`` (N, 3) the unit normal of the surface."""

    name: str
    xy: np.ndarray
    u_mm: np.ndarray
    normal3d: np.ndarray


# -- crossing a grid with a plane ---------------------------------------------------------------

def _oriented(x, y, z, valid):
    """The grids as float arrays with z growing along the rows, a mask of the usable points, and
    whether they had to be transposed (z varying more across the columns than down the rows)."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    valid = np.asarray(valid, dtype=bool)
    if x.ndim != 2 or not (x.shape == y.shape == z.shape == valid.shape):
        raise ValueError(f"x, y, z and valid must be 2D grids of one shape, not {x.shape}, {y.shape}, "
                         f"{z.shape}, {valid.shape}")
    usable = valid & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    z = np.where(usable, z, 0.0)             # nothing non-finite is left to subtract below

    def mean_step(a, b, ok):
        n = np.count_nonzero(ok)
        return float(np.where(ok, np.abs(a - b), 0.0).sum() / n) if n else 0.0

    down = mean_step(z[1:], z[:-1], usable[1:] & usable[:-1])
    across = mean_step(z[:, 1:], z[:, :-1], usable[:, 1:] & usable[:, :-1])
    if across > down:
        return x.T, y.T, z.T, usable.T, True
    return x, y, z, usable, False


def _column_derivative(x, y, z, usable, rows, cols):
    """d(point)/d(column) at the grid points (rows[i], cols[i]): central where both neighbouring columns
    are usable, one-sided where only one is, NaN where none is."""
    n_cols = x.shape[1]
    left, right = np.maximum(cols - 1, 0), np.minimum(cols + 1, n_cols - 1)
    has_left = (cols > 0) & usable[rows, left]
    has_right = (cols < n_cols - 1) & usable[rows, right]
    point = np.stack([x[rows, cols], y[rows, cols], z[rows, cols]], axis=1)
    before = np.stack([x[rows, left], y[rows, left], z[rows, left]], axis=1)
    after = np.stack([x[rows, right], y[rows, right], z[rows, right]], axis=1)
    derivative = np.full(point.shape, np.nan)
    both = has_left & has_right
    derivative[both] = (after - before)[both] / 2.0
    only_right = has_right & ~has_left
    derivative[only_right] = (after - point)[only_right]
    only_left = has_left & ~has_right
    derivative[only_left] = (point - before)[only_left]
    return derivative


def _cut(x, y, z, usable, z0, u_step_mm):
    """``curve_at_z`` on grids that are already oriented (z along the rows)."""
    n_rows, n_cols = z.shape
    u_mm = np.arange(n_cols) * float(u_step_mm)
    xy = np.full((n_cols, 2), np.nan)
    normal = np.full((n_cols, 3), np.nan)
    if n_rows < 2 or n_cols < 1:
        return xy, u_mm, normal
    below = z <= z0
    # min <= z0 < max  <=>  exactly one of the two ends is <= z0 ; only pairs of usable points count
    hit = (below[:-1] != below[1:]) & usable[:-1] & usable[1:]
    cols = np.flatnonzero(hit.any(axis=0))
    if cols.size == 0:
        return xy, u_mm, normal
    rows = hit.argmax(axis=0)[cols]                           # first crossing pair of each column
    a = np.stack([x[rows, cols], y[rows, cols], z[rows, cols]], axis=1)
    b = np.stack([x[rows + 1, cols], y[rows + 1, cols], z[rows + 1, cols]], axis=1)
    t = ((z0 - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]       # the ends differ in z: one is <= z0, the other is not
    point = a + t * (b - a)
    along_rows = b - a
    along_cols = ((1 - t) * _column_derivative(x, y, z, usable, rows, cols)
                  + t * _column_derivative(x, y, z, usable, rows + 1, cols))
    defined = np.isfinite(along_cols).all(axis=1)             # a missing neighbour leaves no normal (and no point)
    cols, point, along_rows, along_cols = cols[defined], point[defined], along_rows[defined], along_cols[defined]
    cross = np.cross(along_cols, along_rows)
    length = np.linalg.norm(cross, axis=1)
    scale = np.linalg.norm(along_cols, axis=1) * np.linalg.norm(along_rows, axis=1)
    good = length > 1e-9 * scale                              # parallel directions: no normal (guards 0 / 0)
    xy[cols[good]] = point[good, :2]
    normal[cols[good]] = cross[good] / length[good, None]
    return xy, u_mm, normal


def curve_at_z(x, y, z, valid, z0: float, u_step_mm: float):
    """Cut a surface grid with the plane z = z0: ``(xy (N, 2), u_mm (N,), normal3d (N, 3))``, one row per column.

    If z varies more across the columns than down the rows the grids are transposed first.  In each
    column the crossing is the first pair of rows (r, r + 1), both usable, with ``min <= z0 < max``;
    the point is interpolated between them.  ``u_mm = column * u_step_mm``, defined for every column.
    The normal is the cross product of the derivatives of the grid along the columns (interpolated
    between the two rows) and along the rows, normalised; its sign depends on the grid orientation but
    is the same all along a curve.  A column without a crossing, or whose crossing has no usable
    neighbour column to differentiate with, gives NaN in ``xy`` and in ``normal3d``.
    """
    x, y, z, usable, _ = _oriented(x, y, z, valid)
    return _cut(x, y, z, usable, float(z0), u_step_mm)


# -- reading the segments of a slice ------------------------------------------------------------

def _read_scale(folder: Path) -> tuple[float, float]:
    """``scale`` of ``<folder>/meta.json``: the grid has one point every ``1 / scale`` voxels."""
    path = folder / "meta.json"
    if not path.is_file():
        raise ValueError(f"{folder}: meta.json is missing (its scale gives the step of the grid)")
    try:
        scale = json.loads(path.read_text(encoding="utf-8")).get("scale")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"{path}: not a JSON object ({exc})") from exc
    if (not isinstance(scale, list) or len(scale) < 2
            or not all(isinstance(s, (int, float)) and np.isfinite(s) and s > 0 for s in scale[:2])):
        raise ValueError(f"{path}: scale must be a list of two positive numbers, not {scale!r}")
    return float(scale[0]), float(scale[1])


def curves_for_slice(tifxyz_dirs: list[Path], names: list[str], sl: SliceImage, cfg: DataConfig) -> list[Curve]:
    """The curves where the segments cross the plane of ``sl``, in pixels of its window.

    The coordinates of each folder are multiplied by ``cfg.tifxyz_to_l0``, cut at ``sl.z_l0``, moved to
    the level of the slice (``l0_to_level``) and shifted by ``sl.origin_xy``.  ``u_mm`` counts the grid
    step along the columns as ``cfg.voxel_mm * cfg.tifxyz_to_l0 / scale[0]`` of ``meta.json``
    (``scale[1]`` if the grid is stored transposed): the grid has one point every ``1 / scale`` tifxyz
    units, a tifxyz unit is ``tifxyz_to_l0`` level-0 voxels (the factor that is applied to the
    coordinates), and a level-0 voxel is ``voxel_mm`` long.  With ``tifxyz_to_l0 = 1.0``, the value
    for the data of wave 1, this is the plain ``cfg.voxel_mm / scale``.
    """
    if len(tifxyz_dirs) != len(names):
        raise ValueError(f"{len(tifxyz_dirs)} folders but {len(names)} names")
    factor = float(cfg.tifxyz_to_l0)
    if not factor > 0:
        raise ValueError(f"tifxyz_to_l0 must be positive, not {cfg.tifxyz_to_l0!r}")
    origin = np.asarray(sl.origin_xy, dtype=float)
    curves = []
    for folder, name in zip(tifxyz_dirs, names):
        folder = Path(folder)
        scale = _read_scale(folder)
        x, y, z, usable, transposed = _oriented(*read_tifxyz(folder))
        u_step_mm = cfg.voxel_mm * factor / scale[1 if transposed else 0]
        xy, u_mm, normal = _cut(x, y, z, usable, sl.z_l0 / factor, u_step_mm)
        xy = l0_to_level(xy * factor, sl.level, cfg.pyramid) - origin
        curves.append(Curve(name, xy, u_mm, normal))
    return curves


# -- what the curves say about the slice --------------------------------------------------------

def _finite_points(xy) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return xy, np.isfinite(xy).all(axis=1)


def center_from_curves(curves: list[Curve]) -> tuple[float, float]:
    """Centroid of the valid points of the innermost curve (the first one)."""
    if not curves:
        raise ValueError("no curve: there is no centre to compute")
    xy, ok = _finite_points(curves[0].xy)
    if not ok.any():
        raise ValueError(f"curve {curves[0].name!r} has no valid point: no centre to compute")
    cx, cy = xy[ok].mean(axis=0)
    return float(cx), float(cy)


def outward_normals(curve: Curve, center_xy) -> np.ndarray:
    """Unit vectors (N, 2) perpendicular to the curve in the plane of the slice, pointing away from ``center_xy``.

    The tangent is the difference of the neighbouring valid points (central; one-sided at an end or
    next to a hole).  NaN where the point is NaN, has no valid neighbour, has a zero-length tangent, or
    where the normal is perpendicular to the direction to the centre (it cannot be oriented).
    """
    xy, ok = _finite_points(curve.xy)
    n = xy.shape[0]
    out = np.full((n, 2), np.nan)
    if n < 2:
        return out
    has_prev = np.zeros(n, dtype=bool)
    has_prev[1:] = ok[1:] & ok[:-1]
    has_next = np.zeros(n, dtype=bool)
    has_next[:-1] = ok[:-1] & ok[1:]
    safe = np.where(ok[:, None], xy, 0.0)
    prev_xy, next_xy = np.roll(safe, 1, axis=0), np.roll(safe, -1, axis=0)     # the wrapped rows are masked out
    tangent = np.full((n, 2), np.nan)
    both = has_prev & has_next
    tangent[both] = (next_xy - prev_xy)[both]
    only_next = has_next & ~has_prev
    tangent[only_next] = (next_xy - safe)[only_next]
    only_prev = has_prev & ~has_next
    tangent[only_prev] = (safe - prev_xy)[only_prev]
    idx = np.flatnonzero(has_prev | has_next)
    length = np.hypot(tangent[idx, 0], tangent[idx, 1])
    idx, tangent, length = idx[length > 0], tangent[idx[length > 0]], length[length > 0]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / length[:, None]
    side = np.sum(normal * (xy[idx] - np.asarray(center_xy, dtype=float)), axis=1)
    oriented = side != 0
    out[idx[oriented]] = normal[oriented] * np.sign(side[oriented])[:, None]
    return out


def stack_from_curves(curves: list[Curve], center_xy, n_theta: int, spacing_px: float,
                      fold_frac: float = FOLD_FRAC) -> np.ndarray:
    """Radii of the curves along ``n_theta`` rays: (len(curves), n_theta), one row per curve, in order.

    Binning is that of ``io_real.radius_profile``: the angle runs from +x towards +y and bin i is
    centred on the ray 2*pi*i / n_theta (as the rays of ``polar.py``); the radius of a bin is the median
    of the radii of the points of that curve that fall in it, NaN where there is none.  A bin where the
    radii of one curve are further apart than ``fold_frac * spacing_px`` is a fold: NaN as well.
    """
    out = np.full((len(curves), n_theta), np.nan)
    limit = fold_frac * spacing_px
    for i, curve in enumerate(curves):
        xy, ok = _finite_points(curve.xy)
        if not ok.any():
            continue
        dx, dy = xy[ok, 0] - center_xy[0], xy[ok, 1] - center_xy[1]
        # the very expression of radius_profile: a change of binning must be made in both
        bins = np.rint((np.arctan2(dy, dx) % (2 * np.pi)) / (2 * np.pi) * n_theta).astype(int) % n_theta
        radius = np.hypot(dx, dy)
        order = np.lexsort((radius, bins))
        bins, radius = bins[order], radius[order]
        first = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
        count = np.diff(np.r_[first, bins.size])
        median = (radius[first + (count - 1) // 2] + radius[first + count // 2]) / 2.0
        spread = radius[first + count - 1] - radius[first]
        flat = spread <= limit
        out[i, bins[first][flat]] = median[flat]
    return out
