import itertools

import numpy as np
import pytest

from scrollorder.ordered_cut import solve_ordered_surfaces, total_cost


def test_single_surface_follows_the_cheapest_row():
    cost = np.ones((6, 9))
    cost[:, 4] = 0.0
    pos = solve_ordered_surfaces(cost, n_surfaces=1, min_sep=1)
    assert pos.shape == (1, 6)
    assert (pos[0] == 4).all()


def test_smoothness_limits_the_jump_between_neighbouring_columns():
    cost = np.ones((4, 12))
    cost[:2, 2] = 0.0
    cost[2:, 9] = 0.0
    pos = solve_ordered_surfaces(cost, 1, min_sep=1, max_step=1)
    assert np.abs(np.diff(pos[0])).max() <= 1


def test_two_surfaces_never_share_a_single_merged_ridge():
    cost = np.ones((5, 20))
    cost[:, 10] = 0.0  # one visible ridge, but two sheets are known to pass
    pos = solve_ordered_surfaces(cost, 2, min_sep=3)
    assert (pos[1] - pos[0] >= 3).all()
    assert ((pos[0] == 10) | (pos[1] == 10)).all()


def test_surfaces_keep_min_sep_from_both_anchors():
    cost = np.ones((3, 12))
    cost[:, 0] = 0.0   # tempting position glued to the lower anchor
    cost[:, 11] = 0.0  # tempting position glued to the upper anchor
    pos = solve_ordered_surfaces(cost, 2, min_sep=3)
    assert (pos[0] >= 2).all()        # distance to anchor at index -1 is pos+1
    assert (pos[1] <= 12 - 3).all()   # distance to anchor at index n_pos


def test_max_sep_keeps_neighbouring_surfaces_together():
    cost = np.ones((3, 30))
    cost[:, 3] = 0.0
    cost[:, 25] = 0.0
    free = solve_ordered_surfaces(cost, 2, min_sep=2)
    assert (free[1] - free[0] == 22).all()
    tied = solve_ordered_surfaces(cost, 2, min_sep=2, max_sep=6)
    assert (tied[1] - tied[0] <= 6).all()


def _brute_force(cost, n_surfaces, min_sep, max_step):
    n_cols, n_pos = cost.shape
    per_col = []
    for combo in itertools.combinations(range(n_pos), n_surfaces):
        ok = combo[0] + 1 >= min_sep and n_pos - combo[-1] >= min_sep
        ok = ok and all(b - a >= min_sep for a, b in zip(combo, combo[1:]))
        if ok:
            per_col.append(combo)
    best = None
    for choice in itertools.product(per_col, repeat=n_cols):
        arr = np.array(choice).T  # (n_surfaces, n_cols)
        if n_cols > 1 and np.abs(np.diff(arr, axis=1)).max() > max_step:
            continue
        c = sum(cost[col, arr[s, col]] for s in range(n_surfaces) for col in range(n_cols))
        best = c if best is None else min(best, c)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_optimum_on_small_instances(seed):
    rng = np.random.default_rng(seed)
    cost = rng.integers(0, 21, size=(3, 8)).astype(float)
    pos = solve_ordered_surfaces(cost, 2, min_sep=2, max_step=1)
    assert total_cost(cost, pos) == _brute_force(cost, 2, 2, 1)


def test_rejects_more_surfaces_than_fit_between_the_anchors():
    with pytest.raises(ValueError, match="do not fit"):
        solve_ordered_surfaces(np.zeros((3, 5)), n_surfaces=3, min_sep=2)


def test_accepts_one_cost_image_per_surface():
    cost = np.ones((2, 4, 14))
    cost[0, :, 3] = 0.0   # only the first sheet is attracted to row 3
    cost[1, :, 9] = 0.0   # only the second sheet is attracted to row 9
    pos = solve_ordered_surfaces(cost, 2, min_sep=2)
    assert (pos[0] == 3).all() and (pos[1] == 9).all()
    assert total_cost(cost, pos) == 0.0


def test_rejects_a_cost_stack_with_the_wrong_number_of_surfaces():
    with pytest.raises(ValueError, match="one cost image per surface"):
        solve_ordered_surfaces(np.ones((3, 4, 14)), 2, min_sep=2)
