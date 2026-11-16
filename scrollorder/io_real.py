"""Readers for real Vesuvius Challenge data (TIFXYZ surfaces, umbilicus).

Formats follow the villa repository at the time of writing:
* TIFXYZ: lasagna/tifxyz_format.md (x.tif, y.tif, z.tif, meta.json, optional mask.tif;
  points with Z <= 0 are invalid; mask channel 0 < 255 invalidates).
* umbilicus JSON: {"control_points": [{"x":..,"y":..,"z":..}, ...]}, linearly
  interpolated in z, as in spiral-fitting/umbilicus.py.

These readers are tested against synthetic files written to that specification,
NOT against the published datasets: check the first real slice by eye.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.interpolate import interp1d


def read_tifxyz(directory):
    """Return (x, y, z, valid) grids of a TIFXYZ folder."""
    d = Path(directory)
    x, y, z = (np.asarray(tifffile.imread(d / f"{n}.tif"), dtype=np.float64) for n in "xyz")
    valid = z > 0
    if (d / "mask.tif").exists():
        mask = np.asarray(tifffile.imread(d / "mask.tif"))
        if mask.ndim == 3:
            mask = mask[..., 0]
        fy, fx = mask.shape[0] / z.shape[0], mask.shape[1] / z.shape[1]
        if fy >= 1 and fx >= 1 and fy == int(fy) and fx == int(fx):
            fy, fx = int(fy), int(fx)
            keep = (mask >= 255).reshape(z.shape[0], fy, z.shape[1], fx).all(axis=(1, 3))
            valid &= keep
    return x, y, z, valid


def slice_at_z(x, y, z, valid, z0: float) -> np.ndarray:
    """Points (n, 2) as (x, y) where the surface grid crosses the plane z = z0."""
    out = []
    lower = (slice(None, -1), slice(None))
    upper = (slice(1, None), slice(None))
    left = (slice(None), slice(None, -1))
    right = (slice(None), slice(1, None))
    for a, b in ((lower, upper), (left, right)):
        za, zb = z[a], z[b]
        hit = valid[a] & valid[b] & ((za - z0) * (zb - z0) <= 0) & (za != zb)
        t = (z0 - za[hit]) / (zb[hit] - za[hit])
        out.append(np.stack([x[a][hit] + t * (x[b][hit] - x[a][hit]),
                             y[a][hit] + t * (y[b][hit] - y[a][hit])], axis=1))
    return np.concatenate(out, axis=0)


def umbilicus_center(path, z: float, coordinate_scale: float = 1.0) -> tuple[float, float]:
    """(x, y) of the scroll axis at height z; all coordinates multiplied by coordinate_scale."""
    points = sorted(json.loads(Path(path).read_text())["control_points"], key=lambda p: p["z"])
    zs = np.array([p["z"] for p in points], dtype=float) * coordinate_scale
    xy = np.array([(p["x"], p["y"]) for p in points], dtype=float) * coordinate_scale
    cx, cy = interp1d(zs, xy, axis=0, fill_value="extrapolate")(z)
    return float(cx), float(cy)


def radius_profile(xy, center_xy, n_theta: int) -> np.ndarray:
    """Median radius of the points in each angular bin (NaN where there is none)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    dx, dy = xy[:, 0] - center_xy[0], xy[:, 1] - center_xy[1]
    bins = np.rint((np.arctan2(dy, dx) % (2 * np.pi)) / (2 * np.pi) * n_theta).astype(int) % n_theta
    radius = np.hypot(dx, dy)
    profile = np.full(n_theta, np.nan)
    for b in np.unique(bins):
        profile[b] = np.median(radius[bins == b])
    return profile


def build_stack(tifxyz_dirs, z: float, center_xy, n_theta: int, xyz_scale: float = 1.0) -> np.ndarray:
    """Stack (n_windings, n_theta) of radii at height z, innermost winding first.

    `z` and `center_xy` are in the coordinates of the image you will analyse;
    `xyz_scale` converts TIFXYZ coordinates to them (0.5 for pyramid level 1, ...).
    """
    rows = []
    for d in tifxyz_dirs:
        x, y, zz, valid = read_tifxyz(d)
        xy = slice_at_z(x * xyz_scale, y * xyz_scale, zz * xyz_scale, valid, z)
        rows.append(radius_profile(xy, center_xy, n_theta))
    rows = [r for r in rows if np.isfinite(r).any()]
    rows.sort(key=lambda r: np.nanmedian(r))
    return np.array(rows)
