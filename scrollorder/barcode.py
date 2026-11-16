"""Integer shift between two stacks of sheets, from their spacing "barcode".

Sheets never cross.  So when n sheets enter an unreadable zone and come out the
other side, the only freedom in how they reconnect is a single integer: the
shift.  `left` and `right` are the sorted radii of the sheets on either side of
the zone; the convention is

    right[j] is the same sheet as left[j + shift]

Radii drift and stacks get compressed across such a zone, so matching by
nearest radius fails.  The irregular spacing between consecutive sheets,
however, travels with the sheets -- like tree-ring widths used for cross-dating
in dendrochronology.  For every candidate shift we fit the best affine map
(offset + uniform compression) between the paired radii and score the residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CONTINUITY_WEIGHT = 0.02   # tie-breaker only: prefer the smaller radial offset
_RESIDUAL_FLOOR = 0.02      # residuals below 2 % of a spacing count as zero


@dataclass
class ShiftEstimate:
    shift: int
    confidence: float           # 0 = barcode says nothing, 1 = unambiguous
    residuals: dict[int, float]  # per candidate shift, in units of the median spacing
    n_pairs: int


def _nearest_shift(left: np.ndarray, right: np.ndarray) -> int:
    nearest = np.abs(right[:, None] - left[None, :]).argmin(axis=1)
    return int(np.rint(np.median(nearest - np.arange(right.size))))


def estimate_shift(left, right, max_shift: int = 3, method: str = "barcode", min_pairs: int = 6) -> ShiftEstimate:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if min(left.size, right.size) < min_pairs:
        raise ValueError(f"need at least {min_pairs} sheets on each side")
    if method == "nearest":
        return ShiftEstimate(_nearest_shift(left, right), 0.0, {}, min(left.size, right.size))
    if method != "barcode":
        raise ValueError(f"unknown method {method!r}")

    spacing = float(np.median(np.diff(left)))
    residuals, offsets, pairs = {}, {}, {}
    for s in range(-max_shift, max_shift + 1):
        j = np.arange(max(0, -s), min(right.size, left.size - s))
        if j.size < min_pairs:
            continue
        x, y = right[j], left[j + s]
        slope, intercept = np.polyfit(x, y, 1)
        residuals[s] = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)) / spacing)
        offsets[s] = float(abs(np.mean(y - x)) / spacing)
        pairs[s] = int(j.size)
    if not residuals:
        raise ValueError(f"need at least {min_pairs} overlapping sheets for some shift")

    best = min(residuals, key=lambda s: residuals[s] + _CONTINUITY_WEIGHT * offsets[s])
    others = [r for s, r in residuals.items() if s != best]
    confidence = 0.0
    if others:
        confidence = 1.0 - (residuals[best] + _RESIDUAL_FLOOR) / (min(others) + _RESIDUAL_FLOOR)
    return ShiftEstimate(best, float(np.clip(confidence, 0.0, 1.0)), residuals, pairs[best])
