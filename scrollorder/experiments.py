"""The two measurements of the protocol (see README).

Both take a `stack` (n_windings, n_theta) of ground-truth radii -- from verified
surfaces on real data -- and hide part of it:

* evaluate_ordered_cut: keep two anchor windings K turns apart, hide the K-1
  sheets between them, recover them from the surface-probability image.
* evaluate_barcode: hide an angular band, and the number of sheets lost at the
  edges of the stack, then recover the integer shift across the band.
"""
from __future__ import annotations

import numpy as np

from .barcode import estimate_shift
from .ordered_cut import solve_ordered_surfaces
from .polar import positions_to_radius, ridge_evidence, sample_between


def wrap_accuracy(pred_radius, truth) -> np.ndarray:
    """Boolean (S, C): is each predicted sheet closest to its own true wrap?

    `truth` is (S + 2, C): lower anchor, the S true sheets, upper anchor.
    """
    pred_radius, truth = np.asarray(pred_radius), np.asarray(truth)
    nearest = np.abs(pred_radius[:, None, :] - truth[None, :, :]).argmin(axis=1)
    return nearest == np.arange(1, pred_radius.shape[0] + 1)[:, None]


def _runs(mask, min_len: int) -> list[np.ndarray]:
    """Index arrays of the contiguous True runs of `mask` that are at least min_len long."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if r.size >= min_len]


def evaluate_ordered_cut(
    polar,
    dr: float,
    stack,
    k_values,
    theta_stride: int = 1,
    anchor_stride: int | None = None,
    pos_per_gap: int = 8,
    alpha: float = 0.35,
    max_step: int = 1,
    min_run: int = 16,
    prior_weight: float = 0.05,
) -> list[dict]:
    """Accuracy of three ways of placing the K-1 sheets between two known windings.

    interp: equal division between the anchors (the baseline to beat);
    snap:   equal division, then each sheet moves to the strongest response nearby;
    cut:    all sheets solved together by minimum cut (order and separation enforced).
            Its cost is (1 - probability) + prior_weight * distance to the equal-division
            position, in units of the mean sheet spacing: where the image shows
            nothing, the sheets fall back to equal division.

    `polar` is the surface-probability slice resampled by `polar_resample` with
    radial step `dr`.  Returns one row per (K, method) with: accuracy (share of
    samples on the correct wrap), accuracy_dense (same, on the 10 % of samples
    with the tightest true spacing), merged (share of neighbouring sheets placed
    on top of each other) and n_samples.
    """
    stack = np.asarray(stack, dtype=float)
    cols_all = np.arange(0, stack.shape[1], theta_stride)
    rows = []
    for k in k_values:
        n_surf, n_pos = k - 1, k * pos_per_gap - 1
        min_sep = max(1, int(alpha * pos_per_gap))
        base = (np.arange(n_surf) + 1) * pos_per_gap - 1          # equal division, on the grid
        half = pos_per_gap // 2
        prior = prior_weight * np.abs(np.arange(n_pos)[None, :] - base[:, None]) / pos_per_gap  # (S, n_pos)
        correct = {m: [] for m in ("interp", "snap", "cut")}
        merged = {m: [] for m in correct}
        spacing = []
        for a in range(0, stack.shape[0] - k, anchor_stride or k):
            truth_all = stack[a : a + k + 1][:, cols_all]
            for run in _runs(np.isfinite(truth_all).all(axis=0), min_run):
                truth = truth_all[:, run]
                profile = sample_between(polar[cols_all[run]], dr, truth[0], truth[-1], n_pos, with_anchors=True)
                prob = ridge_evidence(profile)
                cost = 1.0 - np.clip(prob, 0.0, 1.0)
                window = np.stack([cost[:, b - half : b + half + 1] for b in base])  # (S, C, 2*half+1)
                preds = {
                    "interp": np.broadcast_to(base[:, None], (n_surf, run.size)),
                    "snap": base[:, None] - half + window.argmin(axis=2),
                    "cut": solve_ordered_surfaces(cost[None] + prior[:, None, :], n_surf, min_sep, max_step=max_step),
                }
                for name, pos in preds.items():
                    radius = positions_to_radius(pos, truth[0], truth[-1], n_pos)
                    correct[name].append(wrap_accuracy(radius, truth).ravel())
                    if n_surf > 1:
                        merged[name].append((np.diff(pos, axis=0) <= 0).ravel())
                gaps = np.diff(truth, axis=0)
                spacing.append(np.minimum(gaps[:-1], gaps[1:]).ravel())
        if not spacing:
            continue
        spacing = np.concatenate(spacing)
        dense = spacing <= np.quantile(spacing, 0.1)
        for name in correct:
            ok = np.concatenate(correct[name])
            rows.append({
                "K": int(k),
                "method": name,
                "accuracy": float(ok.mean()),
                "accuracy_dense": float(ok[dense].mean()),
                "merged": float(np.concatenate(merged[name]).mean()) if merged[name] else 0.0,
                "n_samples": int(ok.size),
            })
    return rows


def evaluate_barcode(
    stack,
    band_cols,
    n_trials: int = 200,
    max_drop: int = 3,
    min_sheets: int = 8,
    confident: float = 0.5,
    seed: int = 0,
) -> list[dict]:
    """Accuracy of the integer shift recovered across a hidden angular band.

    For each band width (in columns of `stack`), pick a random ray i and the ray
    i + band, drop a random, unknown number (0..max_drop) of sheets at the inner
    and outer edge of each side, and ask for the shift.  Returns one row per
    (band, method): accuracy, and for the barcode the share of trials above the
    `confident` threshold and the accuracy among those.
    """
    stack = np.asarray(stack, dtype=float)
    n_theta = stack.shape[1]
    rng = np.random.default_rng(seed)
    rows = []
    for band in band_cols:
        hits = {"nearest": [], "barcode": []}
        confidence = []
        attempts = 0
        while len(confidence) < n_trials and attempts < 50 * n_trials:
            attempts += 1
            i = int(rng.integers(0, n_theta - band))
            both = np.isfinite(stack[:, i]) & np.isfinite(stack[:, i + band])
            runs = _runs(both, min_sheets + 2 * max_drop)
            if not runs:
                continue
            sheets = max(runs, key=len)
            p_in, p_out, q_in, q_out = (int(v) for v in rng.integers(0, max_drop + 1, size=4))
            left = stack[sheets, i][p_in : sheets.size - p_out]
            right = stack[sheets, i + band][q_in : sheets.size - q_out]
            truth = q_in - p_in
            for method in hits:
                est = estimate_shift(left, right, max_shift=max_drop + 1, method=method, min_pairs=min_sheets - 2)
                hits[method].append(est.shift == truth)
                if method == "barcode":
                    confidence.append(est.confidence)
        if not confidence:
            continue
        sure = np.array(confidence) >= confident
        for method, ok in hits.items():
            ok = np.array(ok)
            row = {
                "band_cols": int(band),
                "band_deg": 360.0 * band / n_theta,
                "method": method,
                "accuracy": float(ok.mean()),
                "n_trials": int(ok.size),
            }
            if method == "barcode":
                row["confident_share"] = float(sure.mean())
                row["confident_accuracy"] = float(ok[sure].mean()) if sure.any() else float("nan")
            rows.append(row)
    return rows
