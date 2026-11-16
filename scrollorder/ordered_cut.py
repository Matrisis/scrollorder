"""Globally optimal ordered surfaces between two anchors, by minimum cut.

This is the optimal multi-surface graph search used for retinal layers in OCT
(Li, Wu, Chen, Sonka, IEEE TPAMI 2006), applied to the sheets of a scroll.

Setting: `cost[c, m]` is a 2D image in which column `c` runs along the sheet
(e.g. the angle around the scroll axis) and `m` runs across the sheets, from one
anchor winding (virtual index -1) to another anchor winding (virtual index
`n_pos`).  We know that exactly `n_surfaces` sheets pass between the anchors.
All of them are solved together under hard constraints:

* order:        pos[s+1, c] - pos[s, c] >= min_sep   (sheets never merge or cross)
* anchors:      pos[0, c] + 1 >= min_sep and n_pos - pos[-1, c] >= min_sep
* smoothness:   |pos[s, c+1] - pos[s, c]| <= max_step
* optionally:   pos[s+1, c] - pos[s, c] <= max_sep

The returned configuration minimises the summed cost over all sheets and
columns among every configuration satisfying the constraints.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

_MAX_SCALE = 10_000
_INT_LIMIT = 2**30


def _per_surface(cost, n_surfaces: int) -> np.ndarray:
    """Return cost as (n_surfaces, n_cols, n_pos); a 2D image is shared by all surfaces."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim == 2:
        return np.broadcast_to(cost, (n_surfaces,) + cost.shape)
    if cost.ndim != 3 or cost.shape[0] != n_surfaces:
        raise ValueError("cost must be (n_cols, n_pos) or one cost image per surface")
    return cost


def total_cost(cost: np.ndarray, positions: np.ndarray) -> float:
    """Sum of `cost` along every surface of `positions` (n_surfaces, n_cols)."""
    cost = _per_surface(cost, len(positions))
    cols = np.arange(cost.shape[1])
    return float(sum(cost[s][cols, row].sum() for s, row in enumerate(positions)))


def solve_ordered_surfaces(
    cost: np.ndarray,
    n_surfaces: int,
    min_sep: int,
    max_sep: int | None = None,
    max_step: int = 1,
) -> np.ndarray:
    """Return integer positions (n_surfaces, n_cols) of the optimal ordered surfaces.

    `cost` is (n_cols, n_pos), shared by all surfaces, or (n_surfaces, n_cols, n_pos)
    to give each surface its own cost (e.g. image term + a prior on its position).
    """
    cost = _per_surface(cost, n_surfaces)
    _, n_cols, n_pos = cost.shape
    if min_sep < 1 or max_step < 0:
        raise ValueError("min_sep must be >= 1 and max_step >= 0")
    if max_sep is not None and max_sep < min_sep:
        raise ValueError("max_sep must be >= min_sep")

    # Every surface s may only live in [lo[s], lo[s] + width): the range left once
    # the surfaces below and above it have taken their minimum separation.
    width = n_pos - (n_surfaces + 1) * min_sep + 2
    if width < 1:
        raise ValueError(
            f"{n_surfaces} surfaces with min_sep={min_sep} do not fit in {n_pos} positions"
        )
    lo = min_sep - 1 + np.arange(n_surfaces) * min_sep

    n_nodes = n_surfaces * n_cols * width
    ids = np.arange(n_nodes).reshape(n_surfaces, n_cols, width)
    source, sink = n_nodes, n_nodes + 1

    # Integer costs (scipy's max-flow needs integer capacities).
    span = cost.max() - cost.min()
    scale = min(_MAX_SCALE, _INT_LIMIT // max(1, 2 * n_nodes))
    if scale < 50:
        raise ValueError("problem too large for the scipy max-flow backend; use fewer columns")
    icost = np.zeros_like(cost, dtype=np.int64)
    if span > 0:
        icost = np.rint((cost - cost.min()) / span * scale).astype(np.int64)

    # Node weights in local coordinates q = m - lo[s]: choosing the closed set
    # {q <= j} in a column costs exactly icost[lo[s] + j] (up to a constant).
    local = np.stack([icost[s, :, l : l + width] for s, l in enumerate(lo)])  # (S, C, width)
    weights = local[:, :, 1:] - local[:, :, :-1]             # for q >= 1

    us, vs, caps = [], [], []

    def add(u, v, cap):
        u, v = np.broadcast_arrays(u, v)
        if u.size:
            us.append(u.ravel())
            vs.append(v.ravel())
            caps.append(np.broadcast_to(cap, u.shape).ravel())

    inf = int(np.abs(weights).sum()) + 1
    upper = ids[:, :, 1:]
    neg, pos = weights < 0, weights > 0
    add(source, upper[neg], -weights[neg])
    add(upper[pos], sink, weights[pos])
    add(source, ids[:, :, 0], inf)   # base nodes always chosen

    # Closure arcs (infinite): "if u is chosen then v is chosen".
    add(ids[:, :, 1:], ids[:, :, :-1], inf)                       # prefix inside a column
    if n_cols > 1 and width > max_step + 1:                       # smoothness, both ways
        add(ids[:, :-1, max_step + 1 :], ids[:, 1:, 1 : width - max_step], inf)
        add(ids[:, 1:, max_step + 1 :], ids[:, :-1, 1 : width - max_step], inf)
    if n_surfaces > 1:
        add(ids[:-1, :, 1:], ids[1:, :, 1:], inf)                 # pos[s+1] - pos[s] >= min_sep
        if max_sep is not None:
            slack = max_sep - min_sep
            if width > slack + 1:                                 # pos[s+1] - pos[s] <= max_sep
                add(ids[1:, :, slack + 1 :], ids[:-1, :, 1 : width - slack], inf)

    graph = coo_matrix(
        (np.concatenate(caps).astype(np.int32), (np.concatenate(us), np.concatenate(vs))),
        shape=(n_nodes + 2, n_nodes + 2),
    ).tocsr()
    flow = maximum_flow(graph, source, sink).flow

    residual = (graph - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reached = np.zeros(n_nodes + 2, dtype=bool)
    reached[breadth_first_order(residual, source, directed=True, return_predecessors=False)] = True

    chosen = reached[ids]                                         # (S, C, width), a prefix per column
    top = width - 1 - np.argmax(chosen[:, :, ::-1], axis=2)
    return top + lo[:, None]
