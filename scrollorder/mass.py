"""Level of the air, mass crossed along the paths, and the mass of each arc of a slice.

A profile is the grey along a path, sampled every half pixel from its start on the curve ``k``.  The
density ``rho`` is the grey minus the air level ``a`` of the slice (a smooth map built from the bottom of
the wide voids), never clipped.  The mass of an ``m``-sheet path is ``c`` times the integral of ``rho`` from
its start to its crossing with the curve ``k + m``, ``c`` being the mean of its inclinations at both ends;
the mass of an arc is the median over its valid paths.  Divided by the mass of one sheet (``sigma_ref``),
it counts the sheets the path goes through, wherever they touch.

The integral is the trapezoid rule on the samples ``s_j = j * step`` up to ``t``, closed by a last partial
trapezoid from the last sample to the exact endpoint ``t`` (``rho`` at ``t`` is interpolated linearly
between the two samples around it).  No sample beyond the one that follows ``t`` is ever read, and the
same estimator is used for every mass of this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from scrollorder.curves import Curve
from scrollorder.paths import Paths, arc_ids, trace_paths
from scrollorder.thresholds import AIR_TILE_MM, M_MAX, MIN_INCLINATION, MIN_VALID_PROFILE_SHARE, VOID_MIN_PX
from scrollorder.volume import SliceImage

STEP = 0.5                      # px between two samples of a profile (half a voxel)
AIR_PERCENTILE = 25.0           # the air of a tile: this percentile of the void levels that fall in it
AIR_MIN_VALUES = 5              # ... if at least this many fall in it; else the value of the nearest such tile
AIR_EDGE_PX = 1.0               # the void level of a profile is looked for in [1 px, t - 1 px]
MOTIFS = ("hole", "self", "order")         # why a profile is invalid; a tie goes to the first one
_AIR_ROWS_PER_CHUNK = 4096      # profiles held at once in the moving means of air_map

ROW_KEYS = ("k", "arc", "m", "theta", "u_mm", "c", "spacing_px", "mass", "band_mass", "n_profiles", "status",
            "clean", "u_path")
ROW_KEYS_EXTENDED = ROW_KEYS + ("motif",)


@dataclass(eq=False)
class SliceMeasure:
    """What ``measure_slice`` measures on one slice.

    ``rows``: one dict per (curve ``k``, arc, ``m``), keys ``ROW_KEYS_EXTENDED``, plain Python values.
    ``profiles[k]`` (for each curve that starts paths, ``k < len(curves) - 1``): ``"grey"`` (P, S) float32,
    the raw grey along each path (NaN outside the image; 1 in control mode), ``"t"`` and ``"c"``
    (P, m_max) the distance to the crossing with the curve ``k + j + 1`` and the inclination of the path
    (NaN where the path is not "ok"), ``"arc"`` (P,) the arc of its start point (-1 outside any arc),
    ``"start_xy"`` and ``"nu"`` (P, 2).  ``air``: the air map (shape of ``img.data``).  ``sigma_ref0``: the
    median mass of the clean ``m = 1`` arcs, ``s_clean`` the median of their spacings (px).
    """

    rows: list[dict]
    profiles: dict[int, dict]
    air: np.ndarray
    sigma_ref0: float
    s_clean: float


# -- profiles and their integral ----------------------------------------------------------------

def _as_xy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros((0, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (P, 2), not {array.shape}")
    return array


def _check_step(step) -> None:
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"step must be a positive length in px, not {step!r}")


def sample_profiles(img: SliceImage, start_xy, nu, n_samples: int, step: float = STEP) -> np.ndarray:
    """The grey of ``img`` along each ray, ``n_samples`` samples every ``step`` px from its start: (P, S) float32.

    Sample ``j`` of ray ``i`` is at ``start_xy[i] + j * step * nu[i]`` (``nu`` is taken as given: unit
    vectors give samples ``step`` px apart).  The coordinates are pixels of the slice's level in the frame
    of the window, i.e. with the origin of the slice already subtracted, as ``curves_for_slice`` gives
    them: ``(x, y)`` is ``img.data[y, x]`` and nothing more is subtracted here.  Bilinear interpolation
    (``map_coordinates``, ``order=1``); NaN outside the pixel centres of the window and for a NaN ray.
    All the samples of all the rays are interpolated in one call.
    """
    start = _as_xy(start_xy, "start_xy")
    direction = _as_xy(nu, "nu")
    if len(start) != len(direction):
        raise ValueError(f"start_xy and nu must have one row per ray, not {start.shape} and {direction.shape}")
    if int(n_samples) != n_samples or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, not {n_samples!r}")
    _check_step(step)
    s = step * np.arange(int(n_samples))
    x = start[:, 0, None] + s[None, :] * direction[:, 0, None]
    y = start[:, 1, None] + s[None, :] * direction[:, 1, None]
    out = np.full(x.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        data = np.asarray(img.data, dtype=np.float32)
        out[ok] = map_coordinates(data, [y[ok], x[ok]], order=1, mode="constant", cval=np.nan, prefilter=False)
    return out


def _cumulative_trapezoid(rho: np.ndarray, step: float) -> np.ndarray:
    """``cum[:, j]``: trapezoid integral of ``rho`` from sample 0 to sample ``j`` (NaN after a NaN sample)."""
    cum = np.zeros(rho.shape)
    if rho.shape[1] > 1:
        cum[:, 1:] = np.cumsum(0.5 * step * (rho[:, 1:] + rho[:, :-1]), axis=1)
    cum[~np.isfinite(rho[:, 0]), 0] = np.nan                  # the later columns are NaN already
    return cum


def _integral_to(rho: np.ndarray, cum: np.ndarray, t: np.ndarray, step: float) -> np.ndarray:
    """Integral of each row of ``rho`` from 0 to ``t`` (P,), trapezoid with the exact endpoint (see module)."""
    n_rows, n_samples = rho.shape
    out = np.full(n_rows, np.nan)
    rows = np.flatnonzero(np.isfinite(t) & (t >= 0))
    if rows.size == 0:
        return out
    pos = t[rows] / step
    last = np.floor(pos).astype(np.intp)                     # the last sample s_J <= t
    frac = pos - last
    partial = frac > 0
    fits = last + partial < n_samples                        # the sample after t, when it is needed, exists
    rows, last, frac, partial = rows[fits], last[fits], frac[fits], partial[fits]
    value = cum[rows, last]
    if partial.any():
        r, j, f = rows[partial], last[partial], frac[partial]
        low, high = rho[r, j], rho[r, j + 1]
        at_t = low + f * (high - low)
        value[partial] = value[partial] + 0.5 * f * step * (low + at_t)
    out[rows] = value
    return out


def integrate_profiles(rho, t, step: float = STEP) -> np.ndarray:
    """Integral of each profile ``rho`` (P, S), sampled every ``step`` px from 0, from 0 to ``t``.

    ``t`` is (P,) or (P, M) (one endpoint per column, the same profile for all of them); the result has
    the shape of ``t``.  Trapezoid rule on the samples ``s_j <= t``, closed by a partial trapezoid from the
    last of them to ``t`` exactly, with ``rho(t)`` interpolated linearly (none when ``t`` is a sample).
    NaN where ``t`` is NaN or negative, where a sample that is used is NaN, or where ``t`` is beyond the
    samples (the sample after ``t`` is needed but does not exist).
    """
    _check_step(step)
    rho = np.asarray(rho, dtype=float)
    t = np.asarray(t, dtype=float)
    if rho.ndim != 2 or t.ndim not in (1, 2) or t.shape[0] != rho.shape[0]:
        raise ValueError(f"rho must be (P, S) and t (P,) or (P, M), not {rho.shape} and {t.shape}")
    if rho.shape[1] == 0:
        return np.full(t.shape, np.nan)
    cum = _cumulative_trapezoid(rho, step)
    if t.ndim == 1:
        return _integral_to(rho, cum, t, step)
    if t.shape[1] == 0:
        return np.zeros(t.shape)
    return np.stack([_integral_to(rho, cum, t[:, j], step) for j in range(t.shape[1])], axis=1).reshape(t.shape)


def _needed_sample(t: np.ndarray, step: float) -> np.ndarray:
    """Index of the last sample the integral to ``t`` reads (-1 where ``t`` is NaN): J, or J + 1 if t > s_J."""
    out = np.full(t.shape, -1, dtype=np.intp)
    ok = np.isfinite(t) & (t >= 0)
    pos = t[ok] / step
    last = np.floor(pos).astype(np.intp)
    out[ok] = last + (pos > last)
    return out


def _first_bad_sample(grey: np.ndarray) -> np.ndarray:
    """Index of the first non-finite sample of each profile, ``S`` if there is none."""
    bad = ~np.isfinite(grey)
    return np.where(bad.any(axis=1), bad.argmax(axis=1), grey.shape[1])


# -- the air ------------------------------------------------------------------------------------

def _void_levels(grey, t1, start, nu, step, void_min_px):
    """Level and position of the void of each profile: the lowest mean of ``void_min_px`` px of samples that
    lies inside ``[1 px, t1 - 1 px]``; ``(values (V,), xy (V, 2))`` for the profiles that have room for it."""
    width = max(1, int(round(void_min_px / step)))
    values, places = [], []
    for lo in range(0, len(grey), _AIR_ROWS_PER_CHUNK):
        g = np.asarray(grey[lo:lo + _AIR_ROWS_PER_CHUNK], dtype=float)
        t = t1[lo:lo + _AIR_ROWS_PER_CHUNK]
        n_win = g.shape[1] - width + 1
        if n_win < 1:
            continue
        finite = np.isfinite(g)
        zero = np.concatenate([np.zeros((len(g), 1)), np.cumsum(np.where(finite, g, 0.0), axis=1)], axis=1)
        bad = np.concatenate([np.zeros((len(g), 1)), np.cumsum(~finite, axis=1)], axis=1)
        mean = (zero[:, width:] - zero[:, :n_win]) / width                   # window a: samples a .. a + width - 1
        clean = (bad[:, width:] - bad[:, :n_win]) == 0
        first = step * np.arange(n_win)
        lastpos = first + step * (width - 1)
        t_ok = np.where(np.isfinite(t), t, -np.inf)
        allowed = clean & (first[None, :] >= AIR_EDGE_PX) & (lastpos[None, :] <= t_ok[:, None] - AIR_EDGE_PX)
        has = allowed.any(axis=1)
        if not has.any():
            continue
        masked = np.where(allowed, mean, np.inf)
        best = masked.argmin(axis=1)
        rows = np.flatnonzero(has)
        centre = step * (best[rows] + (width - 1) / 2.0)
        values.append(masked[rows, best[rows]])
        places.append(start[lo + rows] + centre[:, None] * nu[lo + rows])
    if not values:
        return np.zeros(0), np.zeros((0, 2))
    return np.concatenate(values), np.concatenate(places)


def _bilinear_weights(n_pixels: int, n_tiles: int, tile_px: float) -> np.ndarray:
    """(n_pixels, n_tiles) weights of linear interpolation between tile centres ((i + 0.5) * tile_px),
    constant beyond the first and last centres."""
    pos = np.clip(np.arange(n_pixels) / tile_px - 0.5, 0.0, n_tiles - 1)
    low = np.minimum(np.floor(pos).astype(np.intp), max(n_tiles - 2, 0))
    frac = pos - low
    weights = np.zeros((n_pixels, n_tiles), dtype=np.float32)
    weights[np.arange(n_pixels), low] = 1.0 - frac
    if n_tiles > 1:
        weights[np.arange(n_pixels), low + 1] += frac
    return weights


def air_map(img: SliceImage, grey, t1, start_xy, nu, tile_px: float, step: float = STEP,
            void_min_px: int = VOID_MIN_PX) -> np.ndarray:
    """The air level of the slice, a smooth map of the shape of ``img.data`` (float32).

    From the ``m = 1`` profiles ``grey`` (Q, S) of the clean arcs, their lengths ``t1`` (Q,), starts and
    directions: the void level of a profile is the lowest moving mean of ``void_min_px`` px of samples
    (``void_min_px / step`` samples) among the windows that lie entirely in ``[1 px, t1 - 1 px]`` and hold
    no NaN; it is placed at the middle of that window.  A profile with no such window gives nothing.

    The slice is cut into square tiles of ``tile_px`` (tile ``(i, j)`` holds the pixels ``y // tile_px == i``,
    ``x // tile_px == j``).  A tile with at least 5 void levels takes their 25th percentile; any other tile
    the value of the nearest such tile (distance between tile indices).  The tiles are smoothed by a
    Gaussian of one tile (``mode="nearest"``), then interpolated bilinearly at every pixel between the tile
    centres ``((i + 0.5) * tile_px)``, constant beyond the outer centres.  ``ValueError`` if no tile has 5
    void levels: the air of the slice is then unknown and is not guessed.
    """
    _check_step(step)
    if not (np.isfinite(tile_px) and tile_px > 0):
        raise ValueError(f"tile_px must be a positive length in px, not {tile_px!r}")
    grey = np.asarray(grey)
    t1 = np.asarray(t1, dtype=float).reshape(-1)
    start, direction = _as_xy(start_xy, "start_xy"), _as_xy(nu, "nu")
    if grey.ndim != 2 or not (len(grey) == len(t1) == len(start) == len(direction)):
        raise ValueError(f"grey (Q, S), t1 (Q,), start_xy and nu (Q, 2) must agree, not {grey.shape}, "
                         f"{t1.shape}, {start.shape}, {direction.shape}")
    height, width = np.asarray(img.data).shape
    n_ty, n_tx = int(np.ceil(height / tile_px)), int(np.ceil(width / tile_px))
    values, xy = _void_levels(grey, t1, start, direction, step, void_min_px)
    tile_of = np.zeros(0, dtype=np.intp)
    if values.size:
        ty = np.clip(np.floor(xy[:, 1] / tile_px).astype(np.intp), 0, n_ty - 1)
        tx = np.clip(np.floor(xy[:, 0] / tile_px).astype(np.intp), 0, n_tx - 1)
        tile_of = ty * n_tx + tx
    counts = np.bincount(tile_of, minlength=n_ty * n_tx)
    has = counts >= AIR_MIN_VALUES
    if not has.any():
        raise ValueError(f"no air: no tile of {tile_px:.1f} px holds {AIR_MIN_VALUES} void levels "
                         f"({values.size} profiles of clean arcs had room for a {void_min_px} px window)")
    tiles = np.zeros(n_ty * n_tx)
    order = np.argsort(tile_of, kind="stable")
    sorted_values = values[order]
    starts = np.cumsum(counts) - counts
    for tile in np.flatnonzero(has):
        tiles[tile] = np.percentile(sorted_values[starts[tile]:starts[tile] + counts[tile]], AIR_PERCENTILE)
    tiles = tiles.reshape(n_ty, n_tx)
    _, (iy, ix) = distance_transform_edt(~has.reshape(n_ty, n_tx), return_indices=True)
    tiles = gaussian_filter(tiles[iy, ix], sigma=1.0, mode="nearest")
    rows = _bilinear_weights(height, n_ty, tile_px)
    cols = _bilinear_weights(width, n_tx, tile_px)
    return (rows @ tiles.astype(np.float32)) @ cols.T


# -- the band variant ---------------------------------------------------------------------------

def _polyline_length(xy: np.ndarray) -> float:
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def band_mass(rho_img: np.ndarray, inner_xy, outer_xy) -> float:
    """Sum of ``rho_img`` over the pixel centres inside the polygon ``inner_xy`` then ``outer_xy`` reversed,
    divided by the mean length of the two polylines; NaN if that length is 0 or a point is NaN.

    Only the pixel centres of the polygon's bounding box that lie in the image are tested
    (``matplotlib.path.Path.contains_points``); no inclination correction.
    """
    inner, outer = _as_xy(inner_xy, "inner_xy"), _as_xy(outer_xy, "outer_xy")
    polygon = np.concatenate([inner, outer[::-1]])
    if len(polygon) < 3 or not np.isfinite(polygon).all():
        return float("nan")
    edge = (_polyline_length(inner) + _polyline_length(outer)) / 2.0
    if not edge > 0:
        return float("nan")
    height, width = rho_img.shape
    x0, y0 = (max(int(np.ceil(v)), 0) for v in polygon.min(axis=0))
    x1 = min(int(np.floor(polygon[:, 0].max())), width - 1)
    y1 = min(int(np.floor(polygon[:, 1].max())), height - 1)
    if x0 > x1 or y0 > y1:
        return 0.0
    gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = PolygonPath(polygon).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
    total = rho_img[gy.ravel()[inside], gx.ravel()[inside]].sum(dtype=np.float64)
    return float(total / edge)


# -- grouping by arc ----------------------------------------------------------------------------

def _group_median(group: np.ndarray, value: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of ``value`` per group (``np.median`` convention), NaN for an empty group; values finite."""
    out = np.full(n_groups, np.nan)
    if group.size == 0:
        return out
    order = np.lexsort((value, group))
    g, v = group[order], value[order]
    counts = np.bincount(g, minlength=n_groups)
    first = np.cumsum(counts) - counts
    filled = counts > 0
    lo = first[filled] + (counts[filled] - 1) // 2
    hi = first[filled] + counts[filled] // 2
    out[filled] = (v[lo] + v[hi]) / 2.0
    return out


@dataclass(eq=False)
class _Curve:
    """What ``measure_slice`` keeps of the paths of one curve between its two halves."""

    paths: Paths
    group: np.ndarray        # (P,) dense index of the arc of each path, -1 outside any arc
    arcs: np.ndarray         # (A,) the arc number of each group
    valid: np.ndarray        # (P, m_max) the path is "ok" and its grey is finite up to its end
    c: np.ndarray            # (P, m_max)
    inner: list              # (A,) the points of each arc on the curve k
    rows: list               # the row dicts of this curve, in (m, arc) order
    row_group: list          # the group of each row


# -- one slice ----------------------------------------------------------------------------------

def _paths(curves, k, center_xy, m_max, max_len):
    return trace_paths(curves, k, center_xy, m_max=m_max, max_len=max_len)


def _first_pass_spacing(curves, center_xy, px_mm) -> float:
    """Median of ``t * c`` of the valid 1-sheet paths of all the curves, with rays of 1 mm."""
    length = 1.0 / px_mm
    spacing = []
    for k in range(len(curves) - 1):
        p = _paths(curves, k, center_xy, 1, length)
        ok = p.status[:, 0] == "ok"
        spacing.append(p.t[ok, 0] * (p.c_start[ok] + p.c_end[ok, 0]) / 2.0)
    spacing = np.concatenate(spacing)
    if spacing.size == 0:
        raise ValueError(f"first pass: no path of 1 mm ({length:.1f} px) reaches the next curve, "
                         f"the spacing of the sheets cannot be estimated")
    return float(np.median(spacing))


def _arc_layout(curve: Curve, px_mm: float, center_xy):
    """Arc ids of the points of a curve, and for each arc (sorted) its points and its middle point."""
    ids = arc_ids(curve, px_mm)
    points = np.flatnonzero(ids >= 0)
    arcs, first, counts = np.unique(ids[points], return_index=True, return_counts=True)
    middle = points[first + counts // 2]
    xy = np.asarray(curve.xy, dtype=float).reshape(-1, 2)
    theta = np.arctan2(xy[middle, 1] - center_xy[1], xy[middle, 0] - center_xy[0]) % (2 * np.pi)
    u_mm = np.asarray(curve.u_mm, dtype=float)[middle]
    inner = [xy[points[f:f + n]] for f, n in zip(first, counts)]
    return ids, arcs, theta, u_mm, inner


def _arc_rows(k, m, data: _Curve, theta, u_mm):
    """The rows of curve ``k`` for ``m`` sheets, without masses: status, ``c``, spacing, ``u_path``, motif."""
    p, group, n_arcs = data.paths, data.group, len(data.arcs)
    in_arc = group >= 0
    valid = data.valid[:, m - 1] & in_arc
    total = np.bincount(group[in_arc], minlength=n_arcs)
    n_valid = np.bincount(group[valid], minlength=n_arcs)
    g = group[valid]
    c_med = _group_median(g, data.c[valid, m - 1], n_arcs)
    spacing = _group_median(g, p.t[valid, m - 1] * data.c[valid, m - 1] / m, n_arcs)
    u_path = np.stack([_group_median(g, p.u_path[valid, j], n_arcs) for j in range(m)], axis=1)
    invalid = in_arc & ~valid
    motif_counts = np.stack([np.bincount(group[invalid & (p.status[:, m - 1] == motif)], minlength=n_arcs)
                             for motif in MOTIFS], axis=1)
    rows = []
    for a in np.flatnonzero(total > 0):
        if n_valid[a] < MIN_VALID_PROFILE_SHARE * total[a]:
            status = "short"
        elif c_med[a] < MIN_INCLINATION:
            status = "inclined"
        else:
            status = "ok"
        motif = ""
        if status == "short" and motif_counts[a].max() > 0:
            motif = MOTIFS[int(motif_counts[a].argmax())]
        rows.append(({
            "k": int(k), "arc": int(data.arcs[a]), "m": int(m), "theta": float(theta[a]), "u_mm": float(u_mm[a]),
            "c": float(c_med[a]), "spacing_px": float(spacing[a]) if status == "ok" else float("nan"),
            "mass": float("nan"), "band_mass": float("nan"), "n_profiles": int(n_valid[a]), "status": status,
            "clean": False, "u_path": [float(u) for u in u_path[a]], "motif": motif,
        }, int(a)))
    return rows


def measure_slice(img: SliceImage, curves: list[Curve], center_xy, m_max: int = M_MAX,
                  constant_grey: bool = False) -> SliceMeasure:
    """Mass crossed by the paths of every curve of the slice, per arc, for ``m = 1 .. m_max`` sheets.

    1. First pass: paths of 1 mm (``1 / img.px_mm`` px) through one sheet from every curve; their median
       ``t * c`` is the spacing ``s`` of the slice (``ValueError`` if no path is valid).
    2. Full pass: paths of ``(m_max + 1) * s`` px from every curve ``k < len(curves) - 1`` (``trace_paths``),
       their grey profiles (``sample_profiles``, ``floor(length / 0.5) + 2`` samples).  A profile is valid
       for ``m`` if its path is "ok" in column ``m - 1`` and its grey is finite up to its end.
    3. Rows: one per arc of the curve ``k`` that has paths (``arc_ids``, keyed by the start point of each
       path; a start point outside any arc gives none, an arc none of whose points starts a path neither) and per
       ``m`` with ``k + m <= len(curves) - 1``.  ``c`` and ``spacing_px`` are the medians of ``c`` and of
       ``t * c / m`` over the valid profiles, ``u_path`` the ``m`` medians of their ``u_path`` columns,
       ``theta`` and ``u_mm`` those of the middle point of the arc on the curve ``k``.  The row is
       ``"short"`` if fewer than ``MIN_VALID_PROFILE_SHARE`` of its profiles are valid (``motif``: the most
       frequent of "hole", "self", "order" among the others, "" if none), else ``"inclined"`` if ``c`` is
       below ``MIN_INCLINATION``, else ``"ok"``.  Only the "ok" rows get ``spacing_px``, ``mass`` and
       ``band_mass`` (NaN otherwise).
    4. Clean rows: the "ok" ``m = 1`` rows whose spacing exceeds the median spacing of the "ok" ``m = 1``
       rows of the slice.  The air map comes from their valid profiles (``air_map``, tiles of
       ``AIR_TILE_MM``); ``ValueError`` if there is none.
    5. Masses: ``rho`` = grey minus the air at each sample, profile mass = ``c`` times its integral to
       ``t`` (``integrate_profiles``), row mass = median over the valid profiles; ``band_mass`` on the
       image ``rho`` between the points of the arc on the curve ``k`` and the ends of its valid paths.
       ``sigma_ref0`` and ``s_clean``: medians of the masses and spacings of the clean rows.

    With ``constant_grey`` the grey is 1 wherever it was sampled (NaN stays NaN) and the air is 0: the
    negative control, through the same paths, arcs, estimator and definitions.
    """
    if len(curves) < 2:
        raise ValueError(f"a slice needs at least two curves to measure a sheet, not {len(curves)}")
    if int(m_max) != m_max or m_max < 1:
        raise ValueError(f"m_max must be a positive integer, not {m_max!r}")
    m_max = int(m_max)
    center_xy = (float(center_xy[0]), float(center_xy[1]))
    n_start = len(curves) - 1

    max_len = (m_max + 1) * _first_pass_spacing(curves, center_xy, img.px_mm)
    n_samples = int(np.floor(max_len / STEP)) + 2

    profiles: dict[int, dict] = {}
    per_curve: dict[int, _Curve] = {}
    for k in range(n_start):
        p = _paths(curves, k, center_xy, m_max, max_len)
        ids, arcs, theta, u_mm, inner = _arc_layout(curves[k], img.px_mm, center_xy)
        arc = ids[p.point_index]
        group = np.where(arc >= 0, np.searchsorted(arcs, arc), -1)
        grey = sample_profiles(img, p.start_xy, p.nu, n_samples)
        if constant_grey:
            grey = np.where(np.isfinite(grey), np.float32(1.0), np.float32(np.nan)).astype(np.float32)
        c = (p.c_start[:, None] + p.c_end) / 2.0
        valid = (p.status == "ok") & (_needed_sample(p.t, STEP) < _first_bad_sample(grey)[:, None])
        data = _Curve(p, group, arcs, valid, c, inner, [], [])
        for m in range(1, min(m_max, n_start - k) + 1):
            for row, a in _arc_rows(k, m, data, theta, u_mm):
                data.rows.append(row)
                data.row_group.append(a)
        per_curve[k] = data
        profiles[k] = {"grey": grey, "t": p.t, "c": c, "arc": arc, "start_xy": p.start_xy, "nu": p.nu}

    # clean rows, then the air from their profiles
    ok_m1 = [row for d in per_curve.values() for row in d.rows if row["m"] == 1 and row["status"] == "ok"]
    if not ok_m1:
        raise ValueError("no clean arc: no arc of the slice has a valid 1-sheet measure")
    median_m1 = float(np.median([row["spacing_px"] for row in ok_m1]))
    for row in ok_m1:
        row["clean"] = bool(row["spacing_px"] > median_m1)
    if constant_grey:
        air = np.zeros(np.asarray(img.data).shape, dtype=np.float32)
    else:
        air = _air_of_clean_arcs(img, per_curve, profiles, max_len)

    # masses
    if constant_grey:
        rho_img = np.ones(air.shape, dtype=np.float32)
    else:
        rho_img = np.asarray(img.data, dtype=np.float32) - air
    for k, data in per_curve.items():
        _fill_masses(k, data, profiles[k], air, rho_img, constant_grey)

    rows = [row for k in sorted(per_curve) for row in per_curve[k].rows]
    clean = [row for row in rows if row["clean"]]
    if not clean:
        raise ValueError("no clean arc: no valid 1-sheet arc of the slice is wider than the median spacing")
    sigma_ref0 = float(np.median([row["mass"] for row in clean]))
    s_clean = float(np.median([row["spacing_px"] for row in clean]))
    return SliceMeasure(rows=rows, profiles=profiles, air=air, sigma_ref0=sigma_ref0, s_clean=s_clean)


def _air_of_clean_arcs(img, per_curve, profiles, max_len) -> np.ndarray:
    """``air_map`` on the valid ``m = 1`` profiles of the clean arcs of all the curves."""
    tile_px = AIR_TILE_MM / img.px_mm
    picked = []
    for k, data in per_curve.items():
        clean_groups = [a for row, a in zip(data.rows, data.row_group) if row["clean"]]
        if not clean_groups:
            continue
        keep = data.valid[:, 0] & np.isin(data.group, clean_groups)
        if keep.any():
            picked.append((k, np.flatnonzero(keep)))
    if not picked:
        raise ValueError("no clean arc: the air level of the slice cannot be measured")
    t1 = np.concatenate([profiles[k]["t"][i, 0] for k, i in picked])
    width = min(int(np.ceil(np.max(t1) / STEP)) + 1, profiles[picked[0][0]]["grey"].shape[1])   # up to t1 only
    grey = np.concatenate([profiles[k]["grey"][i, :width] for k, i in picked])
    start = np.concatenate([profiles[k]["start_xy"][i] for k, i in picked])
    nu = np.concatenate([profiles[k]["nu"][i] for k, i in picked])
    return air_map(img, grey, t1, start, nu, tile_px)


def _fill_masses(k, data: _Curve, prof, air, rho_img, constant_grey) -> None:
    """Masses and band masses of the "ok" rows of curve ``k``."""
    p = data.paths
    grey = prof["grey"].astype(float)
    if constant_grey:
        rho = grey
    else:
        s = STEP * np.arange(grey.shape[1])
        x = p.start_xy[:, 0, None] + s[None, :] * p.nu[:, 0, None]
        y = p.start_xy[:, 1, None] + s[None, :] * p.nu[:, 1, None]
        air_at = np.full(grey.shape, np.nan)
        ok = np.isfinite(grey)
        air_at[ok] = map_coordinates(air, [y[ok], x[ok]], order=1, mode="nearest", prefilter=False)
        rho = grey - air_at
    integral = integrate_profiles(rho, p.t, STEP)                 # NaN where t is NaN (invalid columns)
    profile_mass = data.c * integral
    by_m = {}
    for row, a in zip(data.rows, data.row_group):
        if row["status"] != "ok":
            continue
        m = row["m"]
        if m not in by_m:
            valid = data.valid[:, m - 1] & (data.group >= 0)
            by_m[m] = (valid, _group_median(data.group[valid], profile_mass[valid, m - 1], len(data.arcs)))
        valid, medians = by_m[m]
        row["mass"] = float(medians[a])
        members = valid & (data.group == a)
        ends = p.start_xy[members] + p.nu[members] * p.t[members, m - 1][:, None]
        row["band_mass"] = band_mass(rho_img, data.inner[a], ends)
