"""Paths along the normals of a curve: where they cross the next curves, and arcs of a curve.

A path starts on the curve ``k`` of a slice and goes outward along the local normal of that curve (in the
plane of the slice).  Its ``m``-sheet version ends where it first crosses the curve ``k + m``; it is valid
only if the crossings of the curves ``k + 1 .. k + m`` exist, come in increasing order and the curve ``k``
itself is not crossed again before the end.  ``arc_ids`` cuts a curve into arcs of a given length in mm,
the unit on which the measures of the next steps are aggregated.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy.spatial import cKDTree

from scrollorder.curves import Curve, outward_normals
from scrollorder.thresholds import ARC_MM

MIN_HIT_PX = 0.5                # a crossing closer than this to the start is the start itself, not a crossing
_SAMPLE_PX = 2.0                # the rays are sampled this often to find the candidate segments
_MAX_PIECE_PX = 8.0             # ... in a tree of pieces of segments that are at most this long (see _ray_hits)
_S_TOLERANCE = 1e-9             # a ray through a vertex touches the two segments that share it, rounding aside
_MIN_SINE = 1e-9                # a ray this close to parallel to a segment does not cross it
_QUERIES_PER_CHUNK = 100_000    # the queries held at once: each returns the pieces within ~6 px of a point of a ray
_ARC_EPS = 1e-9                 # in arcs: a point that rounding leaves a hair short of an arc boundary opens the next arc


# -- rays and curves ----------------------------------------------------------------------------

def _valid_segments(xy: np.ndarray) -> np.ndarray:
    """Indices i of the segments (i, i + 1) a ray can hit: both ends are finite and the segment is not the
    last one before a hole, nor the first one after it (the neighbour on the far side of each end is finite
    too, except at the two real ends of the array, where there is none)."""
    n = len(xy)
    if n < 2:
        return np.zeros(0, dtype=int)
    finite = np.isfinite(xy).all(axis=1)
    keep = finite[:-1] & finite[1:]
    keep[1:] &= finite[:-2]               # the point before the start of the segment
    keep[:-1] &= finite[2:]               # the point after its end
    return np.flatnonzero(keep)


def _check_max_len(max_len) -> None:
    if not (np.isfinite(max_len) and max_len > MIN_HIT_PX):
        raise ValueError(f"max_len must be a finite length longer than {MIN_HIT_PX} px (a crossing closer than "
                         f"that to the start is never accepted, so no path could be found), not {max_len!r}")


def _as_rows(values, name: str, what: str) -> np.ndarray:
    """``values`` as a float array (P, 2); an empty array of any shape is no row at all."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros((0, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape ({what}, 2), not {array.shape}")
    return array


def ray_hits(starts: np.ndarray, dirs: np.ndarray, curve_xy: np.ndarray, max_len: float) -> tuple[np.ndarray, np.ndarray]:
    """First crossing of each ray with a polyline: ``(t (P,), index (P,))``.

    ``starts`` and ``dirs`` are (P, 2), ``curve_xy`` (N, 2) (an empty array is accepted for any of them),
    ``max_len`` a finite length longer than 0.5 px, else ``ValueError``.

    ``t`` is the distance from the start to the crossing, in pixels (``dirs`` need not be unit vectors: they
    are normalised), ``index`` the fractional index of the crossing along ``curve_xy`` (``i + s`` for a
    point at the fraction ``s`` of the segment ``(i, i + 1)``), where ``u_mm`` and ``normal3d`` can be
    interpolated.  Only crossings with ``0.5 < t <= max_len`` count.  Both are NaN where there is no
    such crossing, and for a ray with a NaN start or direction or a zero direction.

    A segment is a target only if both its ends are finite and it is not the last segment before a hole,
    nor the first one after it: the point beyond each end must be finite too (the two ends of the array
    have none and are no exception).  A refused segment is skipped: the ray goes on and may cross the
    curve further away.  A ray parallel to a segment does not cross it.

    The candidate segments are found with a ``cKDTree`` of the middles of the segments (cut into pieces
    of at most 8 px, so that a long segment does not widen the queries), queried for all the rays at once
    at every 2 px along each ray; the exact ray-segment test is then done on the candidates, with numpy.
    """
    starts = _as_rows(starts, "starts", "P")
    dirs = _as_rows(dirs, "dirs", "P")
    xy = _as_rows(curve_xy, "curve_xy", "N")
    if len(starts) != len(dirs):
        raise ValueError(f"starts and dirs must have one row per ray, not {starts.shape} and {dirs.shape}")
    _check_max_len(max_len)
    return _ray_hits(starts, dirs, xy, float(max_len))


def _ray_hits(starts: np.ndarray, dirs: np.ndarray, xy: np.ndarray, max_len: float) -> tuple[np.ndarray, np.ndarray]:
    """``ray_hits`` on arrays that have been checked; a ``max_len`` of 0.5 px or less finds nothing."""
    n_rays = len(starts)
    t_out = np.full(n_rays, np.nan)
    frac_out = np.full(n_rays, np.nan)
    if n_rays == 0 or not max_len > MIN_HIT_PX:
        return t_out, frac_out

    segs = _valid_segments(xy)
    a, b = xy[segs], xy[segs + 1]
    length = np.hypot(*(b - a).T)
    segs, a, b, length = segs[length > 0], a[length > 0], b[length > 0], length[length > 0]
    norm = np.hypot(dirs[:, 0], dirs[:, 1])
    usable = np.isfinite(starts).all(axis=1) & np.isfinite(dirs).all(axis=1) & (norm > 0)
    rays = np.flatnonzero(usable)
    if segs.size == 0 or rays.size == 0:
        return t_out, frac_out
    unit = dirs[rays] / norm[rays, None]
    origin = starts[rays]
    edge = b - a

    # The tree holds the middles of pieces of the segments, at most _MAX_PIECE_PX long: the query radius is
    # half the longest piece plus 2 px, so a jump of a few hundred px in the curve (a bad column of the mesh)
    # does not make every query return hundreds of segments.  A piece maps back to its segment; the exact
    # test below is done on whole segments only, so the result does not depend on the cut.
    n_piece = np.ceil(length / _MAX_PIECE_PX).astype(np.intp)               # >= 1: the lengths are positive
    parent = np.repeat(np.arange(len(segs)), n_piece)
    rank = np.arange(parent.size) - np.repeat(np.cumsum(n_piece) - n_piece, n_piece)
    middle = a[parent] + ((rank + 0.5) / n_piece[parent])[:, None] * edge[parent]
    tree = cKDTree(middle)
    radius = (length / n_piece).max() / 2.0 + _SAMPLE_PX
    offsets = _SAMPLE_PX * np.arange(int(np.ceil(max_len / _SAMPLE_PX)) + 1)       # the last one reaches max_len
    per_chunk = max(1, _QUERIES_PER_CHUNK // len(offsets))
    for lo in range(0, rays.size, per_chunk):
        chunk = slice(lo, lo + per_chunk)
        ray_xy, ray_dir, ray_id = origin[chunk], unit[chunk], rays[chunk]
        points = ray_xy[:, None, :] + offsets[None, :, None] * ray_dir[:, None, :]
        found = tree.query_ball_point(points.reshape(-1, 2), radius, return_sorted=False)
        counts = np.fromiter(map(len, found), dtype=np.intp, count=len(found))
        pieces = np.fromiter(chain.from_iterable(found), dtype=np.intp, count=int(counts.sum()))
        cand_seg = parent[pieces]
        cand_ray = np.repeat(np.arange(len(found)) // len(offsets), counts)
        pair = np.unique(cand_ray * len(segs) + cand_seg)                          # one test per (ray, segment)
        cand_ray, cand_seg = pair // len(segs), pair % len(segs)

        d, e, w = ray_dir[cand_ray], edge[cand_seg], a[cand_seg] - ray_xy[cand_ray]
        denom = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]                  # |e| * sin(angle between the ray and the segment)
        across = np.abs(denom) > _MIN_SINE * length[cand_seg]
        cand_ray, cand_seg, d, e, w, denom = (v[across] for v in (cand_ray, cand_seg, d, e, w, denom))
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / denom
        s = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / denom
        hit = (t > MIN_HIT_PX) & (t <= max_len) & (s >= -_S_TOLERANCE) & (s <= 1.0 + _S_TOLERANCE)
        cand_ray, cand_seg, t, s = cand_ray[hit], cand_seg[hit], t[hit], s[hit]
        if t.size == 0:
            continue
        order = np.lexsort((t, cand_ray))                              # by ray, then by distance
        cand_ray, cand_seg, t, s = cand_ray[order], cand_seg[order], t[order], s[order]
        first = np.flatnonzero(np.r_[True, cand_ray[1:] != cand_ray[:-1]])
        t_out[ray_id[cand_ray[first]]] = t[first]
        frac_out[ray_id[cand_ray[first]]] = segs[cand_seg[first]] + np.clip(s[first], 0.0, 1.0)
    return t_out, frac_out


def _interpolate(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Rows of ``values`` (N, d) at the fractional indices ``index`` (P,): (P, d), NaN where the index is NaN."""
    out = np.full((len(index), values.shape[1]), np.nan)
    ok = np.isfinite(index)
    if not ok.any():
        return out
    i0 = np.minimum(np.floor(index[ok]).astype(np.intp), len(values) - 2)
    w = (index[ok] - i0)[:, None]
    low = values[i0]
    out[ok] = np.where(w > 0, low + w * (values[i0 + 1] - low), low)       # no 0 * NaN
    return out


# -- the paths of a curve -----------------------------------------------------------------------

@dataclass(eq=False)
class Paths:
    """The paths that start on the curve ``k`` (one per point that has a direction, see ``trace_paths``).

    ``start_xy`` and ``nu`` (P, 2): start and unit direction, ``point_index`` (P,): index of the start point
    on the curve ``k``.  ``c_start`` (P,) is ``|normal3d . (nu, 0)|`` at the start.  Column ``j`` of the
    others describes the path that crosses ``j + 1`` sheets, from the curve ``k`` to the curve ``k + j + 1``:
    ``t[:, j]`` the distance to that crossing, ``c_end[:, j]`` the ``|normal3d . (nu, 0)|`` there,
    ``u_path[:, j]`` the ``u_mm`` of the point where the path enters the sheet ``k + j`` (the start point
    for ``j = 0``, the crossing with the curve ``k + j`` after it), ``status[:, j]`` ``"ok"``, ``"hole"``
    (a crossing is missing), ``"order"`` (the crossings are not met in increasing order) or ``"self"``
    (before ``t[:, j]`` the path crosses a second time a sheet it has entered, the curve ``k`` included: a
    fold or a seam).  A column is invalid as soon as an earlier one is, with the same status.  Where the
    status is not ``"ok"`` the values of the column are NaN.
    """

    k: int
    start_xy: np.ndarray
    nu: np.ndarray
    c_start: np.ndarray
    t: np.ndarray
    c_end: np.ndarray
    u_path: np.ndarray
    status: np.ndarray
    point_index: np.ndarray | None = None          # the first points of the curve, if nothing else is said

    def __post_init__(self):
        if self.point_index is None:
            self.point_index = np.arange(len(self.start_xy))


def _usable_xy(curve: Curve) -> np.ndarray:
    """``curve.xy`` with NaN rows wherever the point is not usable: no ``normal3d`` or no ``u_mm`` to read."""
    xy = np.array(curve.xy, dtype=float).reshape(-1, 2)
    u = np.asarray(curve.u_mm, dtype=float)
    n3 = np.asarray(curve.normal3d, dtype=float)
    if u.shape != (len(xy),) or n3.shape != (len(xy), 3):
        raise ValueError(f"curve {curve.name!r}: xy {np.shape(curve.xy)}, u_mm {u.shape} and normal3d {n3.shape} "
                         f"must have one row per point")
    xy[~(np.isfinite(xy).all(axis=1) & np.isfinite(n3).all(axis=1) & np.isfinite(u))] = np.nan
    return xy


def trace_paths(curves: list[Curve], k: int, center_xy, m_max: int, max_len: float) -> Paths:
    """The paths from the curve ``k`` along its outward normal, through the next ``m_max`` curves.

    One path per point of the curve ``k`` that is usable (finite ``xy``, ``u_mm`` and ``normal3d``) and
    has an outward normal (``curves.outward_normals``).  Column ``j`` reaches the curve ``k + j + 1``
    (first crossing at ``0.5 < t <= max_len``) and is

    - ``"hole"`` if that crossing is missing (no such curve in the list, no crossing, or a crossing in a
      hole of the curve, see ``ray_hits``);
    - ``"self"`` if, for one of the curves ``k + i`` with ``0 <= i <= j``, the ray crosses it a second time
      before ``t[:, j]``: at a distance less than ``t[:, j]`` and more than 0.5 px after its first crossing
      of that curve (for ``i = 0`` the first crossing is the start point itself, so any crossing of the curve
      ``k`` at ``t > 0.5``).  The path then goes through a fold or a seam, and a sheet it has entered would
      be counted twice.  A second crossing of a curve *after* ``t[:, j]`` does not matter to the column ``j``;
    - ``"order"`` if ``t[:, j]`` does not exceed ``t[:, j - 1]`` (the curves are not met in order);
    - ``"ok"`` otherwise.

    When several apply to a column, ``"hole"`` wins over ``"self"``, which wins over ``"order"``.  A column
    is invalid as soon as an earlier one is, for the same reason: a path through ``m`` sheets is valid
    only if the columns ``0 .. m - 1`` are.  ``u_mm`` and ``normal3d`` at a crossing are interpolated
    linearly between the two points of the segment (the normal is normalised again).
    """
    if not 0 <= k < len(curves):
        raise ValueError(f"k must be the index of a curve (0 <= k < {len(curves)}), not {k!r}")
    if int(m_max) != m_max or m_max < 1:
        raise ValueError(f"m_max must be a positive integer, not {m_max!r}")
    _check_max_len(max_len)
    m_max = int(m_max)
    curve = curves[k]
    own_xy = _usable_xy(curve)
    normals = outward_normals(curve, center_xy)
    has_path = np.isfinite(own_xy).all(axis=1) & np.isfinite(normals).all(axis=1)
    point_index = np.flatnonzero(has_path)
    start_xy, nu = own_xy[point_index], normals[point_index]
    n_paths = len(point_index)
    n3 = np.asarray(curve.normal3d, dtype=float)[point_index]
    c_start = np.abs(n3[:, 0] * nu[:, 0] + n3[:, 1] * nu[:, 1])

    t = np.full((n_paths, m_max), np.nan)
    c_end = np.full((n_paths, m_max), np.nan)
    u_path = np.full((n_paths, m_max), np.nan)
    status = np.full((n_paths, m_max), "ok", dtype="<U8")
    reason = np.full(n_paths, "", dtype="<U8")           # why a path is invalid, "" while it is valid
    t_prev = np.zeros(n_paths)
    u_prev = np.asarray(curve.u_mm, dtype=float)[point_index]
    second = np.full(n_paths, np.inf)        # nearest second crossing, from the start, of the sheets entered so far
    if n_paths:
        t_self, _ = _ray_hits(start_xy, nu, own_xy, max_len)     # the curve k: the start point is its first crossing
        second[np.isfinite(t_self)] = t_self[np.isfinite(t_self)]
    for j in range(m_max):
        alive = np.flatnonzero(reason == "")
        if alive.size == 0:
            status[:, j] = reason
            continue
        t_j = np.full(alive.size, np.nan)
        c_j = np.full(alive.size, np.nan)
        u_j = np.full(alive.size, np.nan)
        target_xy = None
        if k + j + 1 < len(curves):
            target = curves[k + j + 1]
            target_xy = _usable_xy(target)
            t_j, frac = _ray_hits(start_xy[alive], nu[alive], target_xy, max_len)
            u_j = _interpolate(np.asarray(target.u_mm, dtype=float)[:, None], frac)[:, 0]
            normal = _interpolate(np.asarray(target.normal3d, dtype=float), frac)
            size = np.linalg.norm(normal, axis=1)
            good = np.isfinite(size) & (size > 0)
            normal[good] /= size[good, None]
            normal[~good] = np.nan
            c_j = np.abs(normal[:, 0] * nu[alive, 0] + normal[:, 1] * nu[alive, 1])
        new = np.full(alive.size, "", dtype="<U8")
        found = np.isfinite(t_j) & np.isfinite(c_j) & np.isfinite(u_j)
        new[~found] = "hole"
        selfy = np.zeros(alive.size, dtype=bool)
        selfy[found] = second[alive[found]] < t_j[found]       # inf: no second crossing, nothing is True
        new[selfy] = "self"
        if j > 0:
            unordered = np.zeros(alive.size, dtype=bool)
            unordered[found] = t_j[found] <= t_prev[alive[found]]
            new[unordered & (new == "")] = "order"
        reason[alive] = new
        status[:, j] = np.where(reason == "", "ok", reason)
        keep = new == ""
        ok = alive[keep]
        t[ok, j], c_end[ok, j] = t_j[keep], c_j[keep]
        u_path[ok, j] = u_prev[ok]
        t_prev[ok] = t_j[keep]
        u_prev[ok] = u_j[keep]
        # the sheet just entered must not be crossed again before the next ones are met: look for its second
        # crossing, from the first one onwards, for the columns that follow
        left = max_len - t_j[keep].min() if ok.size else 0.0
        if target_xy is not None and left > MIN_HIT_PX:
            again, _ = _ray_hits(start_xy[ok] + t_j[keep][:, None] * nu[ok], nu[ok], target_xy, left)
            seen = np.isfinite(again)
            second[ok[seen]] = np.minimum(second[ok[seen]], t_j[keep][seen] + again[seen])
    return Paths(k, start_xy, nu, c_start, t, c_end, u_path, status, point_index)


# -- arcs ---------------------------------------------------------------------------------------

def arc_ids(curve: Curve, px_mm: float, arc_mm: float = ARC_MM) -> np.ndarray:
    """Arc number of each point of the curve (N,), ``-1`` for the points that belong to no arc.

    The curve is cut into runs of consecutive finite points (a NaN row ends a run).  In a run, the
    distance ``d`` (in mm: pixels times ``px_mm``) from its first point along the polyline gives the arc
    ``floor(d / arc_mm)``: the arcs are half-open, ``[i * arc_mm, (i + 1) * arc_mm)`` in accumulated length,
    so the point at exactly ``i * arc_mm`` opens the arc ``i``.  The remainder at the end of the run, shorter
    than ``arc_mm``, belongs to no arc, nor do the runs that are shorter than that: the last point of a run
    that is exactly ``n * arc_mm`` long is that remainder and gets -1 (``_ARC_EPS``, in arcs, keeps rounding
    from moving a point across a boundary).  Arcs are numbered 0, 1, 2 ... along the whole curve.
    """
    if not (np.isfinite(px_mm) and px_mm > 0 and np.isfinite(arc_mm) and arc_mm > 0):
        raise ValueError(f"px_mm and arc_mm must be positive, not {px_mm!r} and {arc_mm!r}")
    xy = np.asarray(curve.xy, dtype=float).reshape(-1, 2)
    n = len(xy)
    ids = np.full(n, -1, dtype=np.int64)
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return ids
    both = ok[1:] & ok[:-1]
    step = np.zeros(n)
    step[1:][both] = np.hypot(*(xy[1:] - xy[:-1])[both].T) * px_mm
    along = np.cumsum(step)
    is_start = ok & ~np.r_[False, ok[:-1]]
    is_end = ok & ~np.r_[ok[1:], False]
    run = np.cumsum(is_start) - 1                              # the run a finite point belongs to
    length = along[is_end] - along[is_start]                   # mm, one per run
    n_arcs = np.floor(length / arc_mm + _ARC_EPS).astype(np.int64)
    first_arc = np.cumsum(n_arcs) - n_arcs                     # numbers go on from run to run
    run = run[ok]
    local = np.floor((along[ok] - along[is_start][run]) / arc_mm + _ARC_EPS).astype(np.int64)
    inside = local < n_arcs[run]
    ids[np.flatnonzero(ok)[inside]] = first_arc[run[inside]] + local[inside]
    return ids
