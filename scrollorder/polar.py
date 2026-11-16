"""Polar views of a scroll cross-section around its axis (the umbilicus).

Conventions: angle theta_i = 2*pi*i/n_theta, measured from +x towards +y in
image coordinates (x = column, y = row); radius r_j = j*dr.  A polar image has
shape (n_theta, n_r).  A "stack" has shape (n_windings, n_theta): stack[k, i]
is the radius of winding k along ray i, increasing with k.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def polar_resample(image, center_xy, n_theta: int, r_max: float, dr: float = 1.0) -> np.ndarray:
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    radius = np.arange(int(round(r_max / dr))) * dr
    x = center_xy[0] + radius[None, :] * np.cos(theta)[:, None]
    y = center_xy[1] + radius[None, :] * np.sin(theta)[:, None]
    return map_coordinates(np.asarray(image, dtype=float), [y, x], order=1, mode="constant", cval=0.0)


def stack_from_unwrapped(r_unwrapped, n_theta: int) -> np.ndarray:
    """Cut a spiral r(phi), sampled n_theta times per turn, into one row per winding."""
    r = np.asarray(r_unwrapped, dtype=float)
    return r[: (r.size // n_theta) * n_theta].reshape(-1, n_theta)


def _fractions(n_pos: int) -> np.ndarray:
    return (np.arange(n_pos) + 1.0) / (n_pos + 1.0)


def sample_between(polar, dr: float, r_lo, r_hi, n_pos: int, with_anchors: bool = False) -> np.ndarray:
    """Sample each ray at n_pos radii evenly spaced strictly between r_lo and r_hi.

    With `with_anchors`, the samples at r_lo and r_hi are added at both ends
    (n_pos + 2 columns), as `ridge_evidence` expects.
    """
    r_lo, r_hi = np.asarray(r_lo, dtype=float), np.asarray(r_hi, dtype=float)
    frac = np.arange(n_pos + 2) / (n_pos + 1.0) if with_anchors else _fractions(n_pos)
    radius = r_lo[:, None] + frac[None, :] * (r_hi - r_lo)[:, None]
    cols = np.broadcast_to(np.arange(radius.shape[0])[:, None], radius.shape)
    return map_coordinates(np.asarray(polar, dtype=float), [cols, radius / dr], order=1, mode="nearest")


def positions_to_radius(positions, r_lo, r_hi, n_pos: int) -> np.ndarray:
    """Convert positions on the `sample_between` grid back to radii."""
    frac = (np.asarray(positions) + 1.0) / (n_pos + 1.0)
    return np.asarray(r_lo)[None, :] + frac * (np.asarray(r_hi) - np.asarray(r_lo))[None, :]


def ridge_evidence(profile) -> np.ndarray:
    """Keep the response only on the crest of each ridge along the ray.

    `profile` is (n_cols, n_pos + 2) from `sample_between(..., with_anchors=True)`;
    the result is (n_cols, n_pos).  A sheet is a ridge of the surface probability:
    only its crest (the middle of a flat blob) counts as evidence.  The flanks of
    a ridge carry none -- otherwise a sheet whose own ridge has a hole gets pulled
    onto the flank of its neighbour -- and neither does the ridge of an anchor,
    which never forms a crest strictly inside the interval.
    """
    profile = np.asarray(profile, dtype=float)
    evidence = np.zeros_like(profile)
    for row, out in zip(profile, evidence):
        crests, _ = find_peaks(row)
        out[crests] = row[crests]
    return evidence[:, 1:-1]
