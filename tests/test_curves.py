import dataclasses
import time

import numpy as np
import pytest

from scrollorder.curves import (Curve, center_from_curves, curve_at_z, curves_for_slice, outward_normals,
                                stack_from_curves)
from scrollorder.dataconf import load_config
from scrollorder.io_real import radius_profile
from scrollorder.phantom import make_volume_phantom
from scrollorder.volume import SliceImage
from tests.fakes import write_fake_config
from tests.helpers import cylinder_grid, write_tifxyz


def test_curve_follows_the_recto_with_increasing_u():
    ph = make_volume_phantom()
    xy, u, n = curve_at_z(*ph["surfaces"][2], z0=6.0, u_step_mm=0.05)
    cx, cy = ph["center_xy"]
    theta = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx) % (2 * np.pi)
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    truth = np.array([ph["recto"](2, t) for t in theta])
    assert np.max(np.abs(r - truth)) < 0.5
    assert np.all(np.diff(u) > 0)
    radial = np.stack([np.cos(theta), np.sin(theta), 0 * theta], axis=1)
    clean = np.abs(((theta - ph["pressed_theta"] + np.pi) % (2 * np.pi)) - np.pi) > 1.2
    assert np.all(np.abs(np.sum(n * radial, axis=1))[clean] > 0.99)

def test_transposed_grid_gives_the_same_curve():
    x, y, z, valid = make_volume_phantom()["surfaces"][1]
    a = curve_at_z(x, y, z, valid, 6.0, 0.05)[0]
    b = curve_at_z(x.T, y.T, z.T, valid.T, 6.0, 0.05)[0]
    np.testing.assert_allclose(a, b)

def test_invalid_columns_become_gaps():
    x, y, z, valid = make_volume_phantom()["surfaces"][1]
    valid = valid.copy(); valid[:, 100:110] = False
    xy = curve_at_z(x, y, z, valid, 6.0, 0.05)[0]
    assert np.isnan(xy[100:110]).all() and np.isfinite(xy[:100]).all()

def test_center_and_outward_normals():
    ph = make_volume_phantom()
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    c = center_from_curves(curves)
    assert np.hypot(c[0] - ph["center_xy"][0], c[1] - ph["center_xy"][1]) < 2.0
    nu = outward_normals(curves[3], c)
    assert np.all(np.sum(nu * (curves[3].xy - np.array(c)), axis=1) > 0)

def test_fold_blanks_the_column():
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    xy = np.stack([100 + 50 * np.cos(theta), 100 + 50 * np.sin(theta)], axis=1)
    fold = np.array([[100 + 58.0, 100.0]])                 # même angle 0, 8 px plus loin
    c = Curve("w", np.vstack([xy, fold]), np.arange(721.0), np.tile([1.0, 0, 0], (721, 1)))
    stack = stack_from_curves([c], (100.0, 100.0), n_theta=360, spacing_px=12.0)
    assert np.isnan(stack[0, 0]) and np.isfinite(stack[0, 1:]).all()


# --- curve_at_z -----------------------------------------------------------------------------

def _ladder(n_rows=6, n_cols=4):
    """Grid whose column c is the vertical line x = c, y = 2c, z = row (a plane z = z0 crosses it at row z0)."""
    cols = np.arange(n_cols, dtype=float)
    x = np.broadcast_to(cols, (n_rows, n_cols)).copy()
    y = 2 * x
    z = np.broadcast_to(np.arange(n_rows, dtype=float)[:, None], (n_rows, n_cols)).copy()
    return x, y, z, np.ones((n_rows, n_cols), bool)


def test_crossing_is_interpolated_between_the_two_rows_of_the_pair():
    x, y, z, valid = _ladder()
    x[2], x[3] = x[2] - 1.0, x[3] + 1.0                    # the pair (2, 3) is 2 apart in x
    xy, u, n = curve_at_z(x, y, z, valid, 2.25, 0.1)
    np.testing.assert_allclose(xy[:, 0], np.arange(4.0) - 1.0 + 0.25 * 2.0)
    np.testing.assert_allclose(xy[:, 1], 2 * np.arange(4.0))
    np.testing.assert_allclose(u, np.arange(4) * 0.1)


def test_a_plane_through_a_row_or_a_decreasing_z_is_one_crossing_and_the_first_wins():
    x, y, z, valid = _ladder(n_rows=9)
    z[5:] = 10.0 - np.arange(5, 9)[:, None] * np.ones((1, 4)) + 1.0     # z: 0 1 2 3 4 6 5 4 3 (goes back down)
    xy_rows = curve_at_z(x, y, z, valid, 3.5, 1.0)[0]
    assert np.isfinite(xy_rows).all()
    z0 = curve_at_z(x, y, z, valid, 2.0, 1.0)[0]            # exactly on row 2
    np.testing.assert_allclose(z0[:, 0], np.arange(4.0))
    # z0 = 5.5 is first crossed between rows 4 and 5 (z 4 -> 6), not on the way back down
    x2 = x.copy(); x2[4] += 100.0
    first = curve_at_z(x2, y, z, valid, 5.5, 1.0)[0]
    np.testing.assert_allclose(first[:, 0], np.arange(4.0) + 100.0 * 0.25)


def test_no_crossing_gives_nan_rows_and_u_stays_defined():
    x, y, z, valid = _ladder()
    for z0 in (-1.0, 5.0, 9.0):                             # below the grid, on its last row, above it
        xy, u, n = curve_at_z(x, y, z, valid, z0, 0.5)
        assert np.isnan(xy).all() and np.isnan(n).all()
        np.testing.assert_allclose(u, np.arange(4) * 0.5)


def test_first_row_crosses_but_not_when_it_is_invalid():
    x, y, z, valid = _ladder()
    assert np.isfinite(curve_at_z(x, y, z, valid, 0.0, 1.0)[0]).all()          # z0 on row 0: pair (0, 1)
    valid[0] = False                                        # as after read_tifxyz on a z = 0 row
    xy, _, n = curve_at_z(x, y, z, valid, 0.0, 1.0)
    assert np.isnan(xy).all() and np.isnan(n).all()
    xy, _, _ = curve_at_z(x, y, z, valid, 0.5, 1.0)         # the pair (0, 1) touches the invalid row
    assert np.isnan(xy).all()
    assert np.isfinite(curve_at_z(x, y, z, valid, 1.0, 1.0)[0]).all()


def test_invalid_points_stored_as_minus_one_do_not_create_crossings():
    x, y, z, valid = _ladder(n_rows=8)
    z[3, 1] = -1.0; x[3, 1] = y[3, 1] = -1.0; valid[3, 1] = False       # real invalid points are stored as -1
    xy = curve_at_z(x, y, z, valid, 2.5, 1.0)[0]
    assert np.isnan(xy[1]).all() and np.isfinite(xy[2:]).all()      # (column 0 has lost its only neighbour at row 3)
    xy = curve_at_z(x, y, z, valid, 5.5, 1.0)[0]            # column 1 is fine away from the hole
    assert np.isfinite(xy).all()


def test_nan_in_the_grid_is_a_gap_not_a_crash():
    x, y, z, valid = _ladder()
    x[2, 0] = np.nan
    xy = curve_at_z(x, y, z, valid, 2.5, 1.0)[0]
    assert np.isnan(xy[0]).all() and np.isfinite(xy[1:]).all()


def test_normal_is_unit_consistent_in_sign_and_finite_next_to_a_gap():
    ph = make_volume_phantom()
    x, y, z, valid = ph["surfaces"][1]
    valid = valid.copy(); valid[:, 100:110] = False
    xy, _, n = curve_at_z(x, y, z, valid, 6.0, 0.05)
    assert np.isnan(n[100:110]).all()
    assert np.isfinite(n[99]).all() and np.isfinite(n[110]).all()        # one-sided differences next to the gap
    ok = np.isfinite(n).all(axis=1)
    np.testing.assert_allclose(np.linalg.norm(n[ok], axis=1), 1.0)
    cx, cy = ph["center_xy"]
    radial = np.stack([xy[ok, 0] - cx, xy[ok, 1] - cy, 0 * xy[ok, 0]], axis=1)
    assert np.all(np.sum(n[ok] * radial, axis=1) > 0)       # same sign everywhere (here: outward)
    nt = curve_at_z(x.T, y.T, z.T, valid.T, 6.0, 0.05)[2]
    np.testing.assert_allclose(n, nt)


def test_a_degenerate_grid_gives_nan_normals_without_warnings():
    x, y, z, valid = _ladder()
    x[:] = 5.0; y[:] = 7.0                                  # every column is the same vertical line: no tangent
    xy, _, n = curve_at_z(x, y, z, valid, 2.5, 1.0)
    assert np.isnan(n).all() and np.isnan(xy).all()         # xy and normal3d are NaN together
    xy, _, n = curve_at_z(*_ladder(n_rows=3, n_cols=1), 1.5, 1.0)       # one column: no neighbour at all
    assert np.isnan(n).all() and np.isnan(xy).all()


def test_input_shapes_are_checked():
    x, y, z, valid = _ladder()
    with pytest.raises(ValueError):
        curve_at_z(x, y, z[:, :3], valid, 2.0, 1.0)
    with pytest.raises(ValueError):
        curve_at_z(x[0], y[0], z[0], valid[0], 2.0, 1.0)


def test_curve_at_z_is_fast_on_a_real_sized_grid():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 1400, 1300
    theta = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    x = np.broadcast_to(3000 + 800 * np.cos(theta), (n_rows, n_cols)).copy()
    y = np.broadcast_to(3000 + 800 * np.sin(theta), (n_rows, n_cols)).copy()
    z = np.broadcast_to(np.arange(n_rows, dtype=float)[:, None] * 20.0, (n_rows, n_cols)).copy()
    valid = rng.random((n_rows, n_cols)) > 0.17
    z[~valid] = -1.0
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        xy, u, n = curve_at_z(x, y, z, valid, 13_000.5, 0.05)
        best = min(best, time.perf_counter() - start)
    # 6 to 9 ms measured; a per-column loop that also builds the normals takes 0.10 to 0.12 s here, so
    # 0.05 s still has 5x of room for a slow machine (best of 3 absorbs a hiccup) and would catch that loop
    assert best < 0.05, f"curve_at_z took {best:.3f} s"
    assert xy.shape == (n_cols, 2) and u.shape == (n_cols,) and n.shape == (n_cols, 3)
    pair_ok = valid[650] & valid[651]                       # z = 13000 on row 650, 13020 on row 651
    found = np.isfinite(xy[:, 0])
    assert not (found & ~pair_ok).any()                     # nothing outside a valid pair (z = -1 makes no crossing)
    assert found.sum() > 0.9 * pair_ok.sum()                # a few lose their normal: no valid neighbour column


# --- center, outward normals, stack ---------------------------------------------------------

def _circle(r, n=360, center=(200.0, 200.0), name="c"):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1)
    return Curve(name, xy, np.arange(n) * 0.1, np.stack([np.cos(th), np.sin(th), 0 * th], 1)), th


def test_center_is_the_mean_of_the_innermost_curve_and_needs_points():
    inner, _ = _circle(30.0, center=(150.0, 120.0))
    inner.xy[5:9] = np.nan
    outer, _ = _circle(60.0, center=(10.0, 10.0))
    cx, cy = center_from_curves([inner, outer])
    assert (cx, cy) == pytest.approx((150.0, 120.0), abs=0.5)      # the 4 missing points pull it by ~0.3
    empty = Curve("e", np.full((4, 2), np.nan), np.arange(4.0), np.full((4, 3), np.nan))
    with pytest.raises(ValueError, match="no valid point"):
        center_from_curves([empty, inner])
    with pytest.raises(ValueError):
        center_from_curves([])


def test_outward_normals_of_a_circle_are_radial_and_unit():
    c, th = _circle(50.0)
    n = outward_normals(c, (200.0, 200.0))
    assert n.shape == (360, 2)
    radial = np.stack([np.cos(th), np.sin(th)], axis=1)
    np.testing.assert_allclose(n[1:-1], radial[1:-1], atol=1e-9)           # central differences: exactly radial
    np.testing.assert_allclose(n[[0, -1]], radial[[0, -1]], atol=0.01)     # one-sided at the two ends (half a step)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)


def test_outward_means_away_from_the_centre_whatever_the_direction_of_travel():
    c, _ = _circle(50.0)
    reversed_curve = Curve("r", c.xy[::-1].copy(), c.u_mm, c.normal3d)
    n = outward_normals(reversed_curve, (200.0, 200.0))
    assert np.all(np.sum(n * (reversed_curve.xy - 200.0), axis=1) > 0)
    inside = outward_normals(c, (260.0, 200.0))             # a centre outside the curve: still away from it
    assert np.all(np.sum(inside * (c.xy - np.array([260.0, 200.0])), axis=1) > 0)


def test_outward_normals_gaps_and_lone_points():
    c, th = _circle(50.0, n=60)
    c.xy[10:14] = np.nan                                    # a gap
    c.xy[30] = np.nan; c.xy[32] = np.nan                    # point 31 has no neighbour left
    n = outward_normals(c, (200.0, 200.0))
    assert np.isnan(n[10:14]).all() and np.isnan(n[30]).all() and np.isnan(n[32]).all() and np.isnan(n[31]).all()
    assert np.isfinite(n[9]).all() and np.isfinite(n[14]).all()          # one-sided next to the gap
    np.testing.assert_allclose(n[9], [np.cos(th[9]), np.sin(th[9])], atol=0.06)
    ok = np.isfinite(n[:, 0])
    np.testing.assert_allclose(np.linalg.norm(n[ok], axis=1), 1.0)


def test_outward_normals_tangent_without_length_is_undefined():
    xy = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 5.0], [20.0, 10.0]])
    c = Curve("d", xy, np.arange(6.0), np.zeros((6, 3)))
    n = outward_normals(c, (0.0, 0.0))                      # points 0-2 coincide; point 3 sees (10, 0) -> (20, 5)
    assert np.isnan(n[0:3]).all()
    assert np.isfinite(n[3:]).all()


def test_stack_has_one_row_per_curve_in_order_and_matches_radius_profile():
    rng = np.random.default_rng(1)
    n_theta = 90
    rows, curves = [], []
    for i, r in enumerate((40.0, 52.0, 64.0)):
        th = rng.uniform(0, 2 * np.pi, 600)
        rad = r + 0.2 * np.cos(3 * th)
        xy = np.stack([100.0 + rad * np.cos(th), 80.0 + rad * np.sin(th)], axis=1)
        xy[::17] = np.nan
        curves.append(Curve(f"c{i}", xy, np.arange(600.0), np.zeros((600, 3))))
        rows.append(radius_profile(xy[np.isfinite(xy[:, 0])], (100.0, 80.0), n_theta))
    blank = Curve("blank", np.full((5, 2), np.nan), np.arange(5.0), np.full((5, 3), np.nan))
    stack = stack_from_curves(curves + [blank], (100.0, 80.0), n_theta, spacing_px=1e9)
    assert stack.shape == (4, n_theta)
    np.testing.assert_allclose(stack[:3], np.array(rows), equal_nan=True)
    assert np.isnan(stack[3]).all()


def test_stack_angle_convention_is_polar_py():
    n_theta = 360
    pts = np.array([[100.0, 110.0], [90.0, 100.0], [100.0, 90.0], [112.0, 100.0]])   # +y, -x, -y, +x
    c = Curve("q", pts, np.arange(4.0), np.zeros((4, 3)))
    stack = stack_from_curves([c], (100.0, 100.0), n_theta, spacing_px=12.0)
    assert stack[0, 90] == pytest.approx(10.0) and stack[0, 180] == pytest.approx(10.0)
    assert stack[0, 270] == pytest.approx(10.0) and stack[0, 0] == pytest.approx(12.0)
    assert np.isfinite(stack).sum() == 4


def test_fold_threshold_follows_fold_frac_and_spacing_and_is_per_curve():
    inner, _ = _circle(50.0, n=360, center=(100.0, 100.0), name="a")
    inner = Curve("a", np.vstack([inner.xy, [[103.0 + 50.0, 100.0]]]), np.arange(361.0), np.zeros((361, 3)))
    other, _ = _circle(62.0, n=360, center=(100.0, 100.0), name="b")           # 12 px further: not a fold
    stack = stack_from_curves([inner, other], (100.0, 100.0), 360, spacing_px=12.0)        # 3 px < 0.3 * 12
    assert np.isfinite(stack).all() and stack[0, 0] == pytest.approx(51.5)
    stack = stack_from_curves([inner, other], (100.0, 100.0), 360, spacing_px=12.0, fold_frac=0.2)
    assert np.isnan(stack[0, 0]) and np.isfinite(stack[0, 1:]).all() and np.isfinite(stack[1]).all()
    stack = stack_from_curves([inner, other], (100.0, 100.0), 360, spacing_px=8.0)         # 3 px > 0.3 * 8
    assert np.isnan(stack[0, 0])


# --- curves_for_slice -----------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return load_config(write_fake_config(tmp_path, "S", 1))        # voxel_mm 0.002399, "mean", tifxyz_to_l0 1.0


def _cylinder():
    """Cylinder of radius 40 around (120, 90), z = 10 to 60 in steps of 2: z0 = 25 crosses between rows 7 and 8."""
    x, y, z = cylinder_grid(center_xy=(120.0, 90.0), radius=40.0, z_range=(10.0, 60.0), n_rows=26, n_cols=181)
    theta = np.linspace(0.0, 2 * np.pi, 181)
    return x, y, z, np.stack([120.0 + 40.0 * np.cos(theta), 90.0 + 40.0 * np.sin(theta)], axis=1)


def _slice(level=0, origin=(0, 0), z_l0=25.0):
    return SliceImage(np.zeros((4, 4), np.float32), origin, level, 0, z_l0, 0.002399 * 2 ** level)


def test_curves_for_slice_reads_the_folders_and_converts_the_grid_step_to_mm(tmp_path, cfg):
    x, y, z, truth = _cylinder()
    a = write_tifxyz(tmp_path / "a", x, y, z, scale=(0.1, 0.05))
    b = write_tifxyz(tmp_path / "b", x.T, y.T, z.T, scale=(0.1, 0.05))          # transposed grid: the other scale
    curves = curves_for_slice([a, b], ["wa", "wb"], _slice(), cfg)
    assert [c.name for c in curves] == ["wa", "wb"] and all(isinstance(c, Curve) for c in curves)
    for c in curves:
        np.testing.assert_allclose(c.xy, truth, atol=1e-4)                       # float32 files
        assert c.normal3d.shape == (181, 3)
    np.testing.assert_allclose(curves[0].u_mm, np.arange(181) * 0.002399 / 0.1)           # scale[0] along columns
    np.testing.assert_allclose(curves[1].u_mm, np.arange(181) * 0.002399 / 0.05)          # scale[1]: transposed grid
    radial = np.stack([np.cos(np.linspace(0, 2 * np.pi, 181)), np.sin(np.linspace(0, 2 * np.pi, 181)), np.zeros(181)], 1)
    assert np.all(np.abs(np.sum(curves[0].normal3d * radial, axis=1)) > 0.999)


def test_curves_for_slice_multiplies_by_tifxyz_to_l0(tmp_path, cfg):
    x, y, z, truth = _cylinder()
    x = x + 3.0 * np.arange(26)[:, None]                  # x grows by 3 per row: the xy tell which pair of rows crosses
    folder = write_tifxyz(tmp_path / "a", x, y, z)
    cfg2 = dataclasses.replace(cfg, tifxyz_to_l0=2.0)                      # one tifxyz unit = 2 level-0 voxels
    for z_l0, row in ((50.0, 7.5), (25.0, 1.25)):          # z = 25 and 12.5 in tifxyz units: rows 7.5 and 1.25
        xy = curves_for_slice([folder], ["w"], _slice(z_l0=z_l0), cfg2)[0].xy
        np.testing.assert_allclose(xy, 2.0 * (truth + [3.0 * row, 0.0]), atol=1e-3)
    same = curves_for_slice([folder], ["w"], _slice(z_l0=25.0), cfg)[0].xy         # factor 1: z = 25 is row 7.5
    np.testing.assert_allclose(same, truth + [3.0 * 7.5, 0.0], atol=1e-3)


def test_curves_for_slice_u_step_in_mm_follows_tifxyz_to_l0_and_the_scale_axis(tmp_path, cfg):
    x, y, z, _ = _cylinder()
    scale = (0.05, 0.1)                                     # not square: which entry is used shows in the step
    straight = write_tifxyz(tmp_path / "a", x, y, z, scale=scale)
    transposed = write_tifxyz(tmp_path / "b", x.T, y.T, z.T, scale=scale)
    cfg2 = dataclasses.replace(cfg, tifxyz_to_l0=2.0)       # a grid step is 1 / scale tifxyz units = 2 / scale voxels
    a, b = curves_for_slice([straight, transposed], ["a", "b"], _slice(z_l0=50.0), cfg2)
    columns = np.arange(181)
    np.testing.assert_allclose(a.u_mm, columns * (0.002399 * 2.0 / 0.05), rtol=1e-12)      # scale[0] along the columns
    np.testing.assert_allclose(b.u_mm, columns * (0.002399 * 2.0 / 0.1), rtol=1e-12)       # scale[1]: transposed grid
    one_a, one_b = curves_for_slice([straight, transposed], ["a", "b"], _slice(z_l0=25.0), cfg)    # factor 1.0
    np.testing.assert_allclose(one_a.u_mm, columns * (0.002399 / 0.05), rtol=1e-12)
    np.testing.assert_allclose(one_b.u_mm, columns * (0.002399 / 0.1), rtol=1e-12)


def test_curves_for_slice_moves_to_the_level_and_the_window(tmp_path, cfg):
    x, y, z, truth = _cylinder()
    folder = write_tifxyz(tmp_path / "a", x, y, z)
    mean = curves_for_slice([folder], ["w"], _slice(level=2, origin=(100, 50)), cfg)[0]
    np.testing.assert_allclose(mean.xy, (truth + 0.5) / 4 - 0.5 - np.array([100.0, 50.0]), atol=1e-4)
    stride = curves_for_slice([folder], ["w"], _slice(level=2, origin=(100, 50)),
                              dataclasses.replace(cfg, pyramid="stride"))[0]
    np.testing.assert_allclose(stride.xy, truth / 4 - np.array([100.0, 50.0]), atol=1e-4)
    shifted = curves_for_slice([folder], ["w"], _slice(origin=(7, -3)), cfg)[0]
    np.testing.assert_allclose(shifted.xy, truth - np.array([7.0, -3.0]), atol=1e-4)
    np.testing.assert_allclose(mean.u_mm, np.arange(181) * 0.002399 / 0.05)       # u does not depend on the level


def test_curves_for_slice_invalid_points_are_gaps(tmp_path, cfg):
    x, y, z, truth = _cylinder()
    x[:, 20:25] = y[:, 20:25] = z[:, 20:25] = -1.0                        # as in the real files
    folder = write_tifxyz(tmp_path / "a", x, y, z)
    c = curves_for_slice([folder], ["w"], _slice(), cfg)[0]
    assert np.isnan(c.xy[20:25]).all() and np.isnan(c.normal3d[20:25]).all()
    assert np.isfinite(c.xy[:20]).all() and np.isfinite(c.xy[25:]).all()
    assert np.isfinite(c.u_mm).all()


def test_curves_for_slice_complains_clearly(tmp_path, cfg):
    x, y, z, _ = _cylinder()
    folder = write_tifxyz(tmp_path / "a", x, y, z)
    with pytest.raises(ValueError, match="names"):
        curves_for_slice([folder], ["a", "b"], _slice(), cfg)
    (folder / "meta.json").unlink()
    with pytest.raises(ValueError, match="meta.json"):
        curves_for_slice([folder], ["a"], _slice(), cfg)
    (folder / "meta.json").write_text('{"format": "tifxyz"}')
    with pytest.raises(ValueError, match="scale"):
        curves_for_slice([folder], ["a"], _slice(), cfg)
