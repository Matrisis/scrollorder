"""Joins between sheets: where the sheet is glued to itself, a path crosses one layer more than it has sheets.

A join is a *run* of ``m = 1`` arcs of one winding ``k`` whose mass is about two sheets (``JOIN_MASS``
times the mass ``sigma_ref`` of one sheet), consecutive along ``u`` (at most one arc missing between two of
them), at least ``JOIN_MIN_MM`` long, and whose neighbouring windings ``k - 1`` and ``k + 1``, over the same
angles, weigh one sheet (``JOIN_NEIGHBOUR_NORMAL``): a heavy neighbour means the mass comes from two
sheets that touch, not from a join.  A join excuses counting errors, so it must rest on evidence: a run is
kept only if at least one neighbour has arcs over its angles and is normal, and none that has arcs is not.
The runs of a slice are then confirmed by the companion slices (``confirm_joins``).

Only the mass of the rows is used: a row takes part if its status is in ``MASS_VALID_STATUSES``
(``"ok"`` or ``"band_nan"``), whatever its band.  Rows with ``m != 1`` are ignored.

``u`` is the position along the grid of the curve (``Curve.u_mm``), taken at the middle of each arc; each
arc stands for ``arc_mm`` of ``u`` around it, so a run goes from half an arc before its first arc to half an
arc after its last.

The angles of a run do not depend on the direction of the grid: nothing says that ``theta`` grows with ``u``
(a grid whose columns run clockwise has it fall).  The ``theta`` of the arcs of a run, taken in ``u`` order,
are unwrapped (``np.unwrap``: two successive arcs never differ by more than pi), and the run covers
``[theta_lo, theta_hi]``, the least and the greatest of them, so ``theta_lo <= theta_hi`` always; the bounds may be
negative or exceed ``2 * pi``.  A neighbouring arc is inside if ``(theta - theta_lo) mod (2 * pi)`` is at most
``theta_hi - theta_lo`` (ends included, up to ``ANGLE_TOL``); an interval of ``2 * pi`` or more takes every angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scrollorder.mass import MASS_VALID_STATUSES
from scrollorder.thresholds import ARC_MM, JOIN_MASS, JOIN_MATCH_MM, JOIN_MIN_MM, JOIN_NEIGHBOUR_NORMAL

MAX_ARC_STEP = 2                # two successive arcs of a run differ by at most 2: at most one arc is missing
ANGLE_TOL = 1e-9                # rad: the ends of an angular interval are included to within the rounding of unwrap
_TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class JoinRun:
    """A run of heavy arcs on winding ``k``: ``u_lo``, ``u_hi`` (mm along the grid, half an arc beyond the
    first and last arc) and ``theta_lo <= theta_hi`` (rad), the least and the greatest angle of its arcs once
    unwrapped along ``u`` (they may be negative or exceed ``2 * pi``)."""

    k: int
    u_lo: float
    u_hi: float
    theta_lo: float
    theta_hi: float


# -- the arguments ------------------------------------------------------------------------------

def _check_range(name: str, bounds) -> tuple[float, float]:
    try:
        lo, hi = (float(b) for b in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (low, high) pair of numbers, not {bounds!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"{name} must be a finite (low, high) pair with low <= high, not {bounds!r}")
    return lo, hi


def _check_length(name: str, value, positive: bool) -> float:
    value = float(value)
    if not (np.isfinite(value) and (value > 0 if positive else value >= 0)):
        raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} length in mm, "
                         f"not {value!r}")
    return value


# -- the runs -----------------------------------------------------------------------------------

@dataclass
class _Arcs:
    """The one-sheet mass-valid arcs of one winding, ordered by arc; ``ratio`` is ``mass / sigma_ref``."""

    arc: np.ndarray
    theta: np.ndarray
    u: np.ndarray
    ratio: np.ndarray


def _arcs_by_winding(rows, sigma_ref: float) -> dict[int, _Arcs]:
    picked: dict[int, list] = {}
    for r in rows:
        if r["m"] != 1 or r["status"] not in MASS_VALID_STATUSES:
            continue
        mass, theta, u = float(r["mass"]), float(r["theta"]), float(r["u_mm"])
        if not (np.isfinite(mass) and np.isfinite(theta) and np.isfinite(u)):
            continue                                        # not reached by construction (mass-valid rows)
        picked.setdefault(int(r["k"]), []).append((int(r["arc"]), theta, u, mass / sigma_ref))
    out = {}
    for k, items in picked.items():
        arc, theta, u, ratio = (np.array(column) for column in zip(*sorted(items, key=lambda it: it[0])))
        out[k] = _Arcs(arc.astype(np.int64), theta, u, ratio)
    return out


def _consecutive_pieces(selected: np.ndarray, arc: np.ndarray) -> list[np.ndarray]:
    """Cut the indices ``selected`` (ascending) where two successive ones are more than ``MAX_ARC_STEP`` arcs apart."""
    if selected.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(arc[selected]) > MAX_ARC_STEP) + 1
    return np.split(selected, cuts)


def _inside_angles(theta: np.ndarray, theta_lo: float, theta_hi: float) -> np.ndarray:
    """Which of the angles ``theta`` (rad, any turn) are in ``[theta_lo, theta_hi]`` modulo ``2 * pi``, ends included
    (to within ``ANGLE_TOL``, whichever side of the turn the rounding falls); every angle if the interval is a turn."""
    offset = (theta - theta_lo) % _TWO_PI                   # in [0, 2 pi): a span of a turn or more takes every angle
    return (offset <= theta_hi - theta_lo + ANGLE_TOL) | (offset >= _TWO_PI - ANGLE_TOL)


def _neighbours_confirm(by_k: dict[int, _Arcs], k: int, theta_lo: float, theta_hi: float, normal) -> bool:
    """No neighbour winding that has arcs over the angles of the run is abnormal, and one at least is normal."""
    one_is_normal = False
    for neighbour in (k - 1, k + 1):
        arcs = by_k.get(neighbour)                          # a winding that does not exist has no arcs
        if arcs is None:
            continue
        inside = arcs.ratio[_inside_angles(arcs.theta, theta_lo, theta_hi)]
        if inside.size == 0:
            continue                                        # no data: it cannot veto, nor validate
        if not normal[0] <= float(np.median(inside)) <= normal[1]:
            return False
        one_is_normal = True
    return one_is_normal


def find_join_runs(rows_m1: list[dict], sigma_ref: float, arc_mm: float = ARC_MM, min_mm: float = JOIN_MIN_MM,
                   mass_range=JOIN_MASS, neighbour_range=JOIN_NEIGHBOUR_NORMAL) -> list[JoinRun]:
    """The join runs of a slice, ordered by winding then by position.

    ``rows_m1``: rows of ``measure_slice`` (the ones with ``m == 1`` and a mass-valid status are used, grouped by
    ``k`` and ordered by ``arc``).  An arc is heavy if ``mass / sigma_ref`` is in ``mass_range`` (bounds included).
    Two successive heavy arcs of a winding are in the same run if their ``arc`` numbers differ by at most 2.
    A run goes from ``u_mm`` of its first arc minus ``arc_mm / 2`` to ``u_mm`` of its last arc plus
    ``arc_mm / 2``, and is kept if that is at least ``min_mm`` long.  It also needs its neighbours: for each of
    the windings ``k - 1`` and ``k + 1`` that has arcs whose ``theta`` is inside ``[theta_lo, theta_hi]``
    (modulo ``2 * pi``; ``theta_lo`` and ``theta_hi`` are the least and greatest angle of the arcs of the run
    unwrapped along ``u``, so the interval is the same whichever way the grid runs), the median of
    ``mass / sigma_ref`` of these arcs must be in ``neighbour_range`` (bounds included); a winding with no such arc
    (or that does not exist) neither vetoes nor validates, but at least one of the two must have arcs and be normal.
    ``ValueError`` for a ``sigma_ref`` that is not a positive number or a range or length that makes no sense.

    Limit: the arcs of a winding are numbered along its grid, which does not wrap.  On a closed ring (the phantom)
    a join that straddles the seam of the grid, the end of arc ``n - 1`` and the start of arc ``0``, is cut in two
    runs, each of which may be shorter than ``min_mm``: it is then missed.  This is the conservative side (a join
    excuses counting errors, a missed one excuses none).
    """
    sigma_ref = float(sigma_ref)
    if not (np.isfinite(sigma_ref) and sigma_ref > 0):
        raise ValueError(f"sigma_ref must be a positive, finite mass, not {sigma_ref!r}")
    arc_mm = _check_length("arc_mm", arc_mm, positive=True)
    min_mm = _check_length("min_mm", min_mm, positive=False)
    heavy_lo, heavy_hi = _check_range("mass_range", mass_range)
    normal = _check_range("neighbour_range", neighbour_range)

    by_k = _arcs_by_winding(rows_m1, sigma_ref)
    runs = []
    for k in sorted(by_k):
        arcs = by_k[k]
        heavy = np.flatnonzero((heavy_lo <= arcs.ratio) & (arcs.ratio <= heavy_hi))
        for piece in _consecutive_pieces(heavy, arcs.arc):
            first, last = int(piece[0]), int(piece[-1])
            u_lo, u_hi = float(arcs.u[first]) - arc_mm / 2.0, float(arcs.u[last]) + arc_mm / 2.0
            if not u_hi - u_lo >= min_mm:
                continue
            turned = np.unwrap(arcs.theta[piece])               # the angles along u, whichever way the grid runs
            theta_lo, theta_hi = float(turned.min()), float(turned.max())
            if _neighbours_confirm(by_k, k, theta_lo, theta_hi, normal):
                runs.append(JoinRun(k, u_lo, u_hi, theta_lo, theta_hi))
    return runs


# -- confirmation by the companion slices -------------------------------------------------------

def confirm_joins(runs: list[JoinRun], companion_runs: list[list[JoinRun]],
                  match_mm: float = JOIN_MATCH_MM) -> list[JoinRun]:
    """The runs that every companion slice also has, in the order of ``runs``.

    ``companion_runs`` holds one list of runs per companion slice.  A run is kept if, for every one of these
    lists, there is a run of the same ``k`` whose ``[u_lo, u_hi]`` meets ``[u_lo - match_mm, u_hi + match_mm]``
    (ends included).  With no companion slice nothing can be confirmed: the result is empty.
    """
    match_mm = _check_length("match_mm", match_mm, positive=False)
    if len(companion_runs) == 0:
        return []
    return [run for run in runs
            if all(any(c.k == run.k and c.u_lo <= run.u_hi + match_mm and c.u_hi >= run.u_lo - match_mm
                       for c in companion)
                   for companion in companion_runs)]


# -- layers -------------------------------------------------------------------------------------

def in_join(k: int, u: float, joins: list[JoinRun]) -> bool:
    """Whether position ``u`` of winding ``k`` is in a join (``u_lo <= u <= u_hi``, ends included)."""
    for join in joins:
        if join.k == k and join.u_lo <= u <= join.u_hi:
            return True
    return False


def expected_layers(k: int, m: int, u_path: list[float], joins: list[JoinRun]) -> int:
    """Layers of papyrus an ``m``-sheet path from winding ``k`` crosses: ``m`` plus one per join it goes through.

    ``u_path[j]`` is where the path enters sheet ``k + j`` (``measure_slice``), so the result is
    ``m + sum(in_join(k + j, u_path[j], joins) for j in range(m))``.  ``ValueError`` if ``u_path`` has fewer than
    ``m`` entries (the extra ones are not looked at) or ``m`` is not a positive integer.
    """
    if int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer, not {m!r}")
    m = int(m)
    if len(u_path) < m:
        raise ValueError(f"u_path has {len(u_path)} positions, a path of {m} sheets needs {m}")
    return m + sum(in_join(int(k) + j, u_path[j], joins) for j in range(m))


def sigma_ref_final(rows_m1: list[dict], joins: list[JoinRun]) -> float:
    """Mass of one sheet: the median mass of the clean ``m = 1`` arcs with a valid mass outside the joins.

    A clean arc is one of the rows with ``clean`` set (``measure_slice``); an arc in a join (``in_join`` at its
    ``u_mm``) is left out, a ``"band_nan"`` arc is kept (its mass is valid).  ``ValueError`` if no arc is left.
    """
    masses = [float(r["mass"]) for r in rows_m1
              if r["m"] == 1 and r["status"] in MASS_VALID_STATUSES and r["clean"]
              and np.isfinite(r["mass"]) and not in_join(r["k"], r["u_mm"], joins)]
    if not masses:
        raise ValueError("no clean one-sheet arc with a valid mass outside the joins: "
                         "the mass of one sheet cannot be measured")
    return float(np.median(masses))
