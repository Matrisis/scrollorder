import json

import numpy as np

from scrollorder.io_real import radius_profile, read_tifxyz, slice_at_z, umbilicus_center
from tests.helpers import cylinder_grid, write_tifxyz

CENTER = (120.0, 90.0)


def test_read_tifxyz_returns_the_stored_grid(tmp_path):
    x, y, z = cylinder_grid()
    rx, ry, rz, valid = read_tifxyz(write_tifxyz(tmp_path / "w001.tifxyz", x, y, z))
    assert np.allclose(rx, x) and np.allclose(ry, y) and np.allclose(rz, z)
    assert valid.all()


def test_read_tifxyz_invalidates_points_with_non_positive_z(tmp_path):
    x, y, z = cylinder_grid()
    x[3, 5], y[3, 5], z[3, 5] = -1.0, -1.0, -1.0
    valid = read_tifxyz(write_tifxyz(tmp_path / "s.tifxyz", x, y, z))[3]
    assert not valid[3, 5] and valid.sum() == valid.size - 1


def test_read_tifxyz_applies_the_validity_mask(tmp_path):
    x, y, z = cylinder_grid()
    mask = np.full(x.shape, 255, dtype=np.uint8)
    mask[:, :10] = 0
    valid = read_tifxyz(write_tifxyz(tmp_path / "s.tifxyz", x, y, z, mask=mask))[3]
    assert not valid[:, :10].any() and valid[:, 10:].all()


def test_slice_at_z_returns_the_intersection_curve(tmp_path):
    x, y, z = cylinder_grid(radius=40.0)
    xy = slice_at_z(x, y, z, np.ones(x.shape, bool), z0=35.5)
    assert xy.shape[0] >= x.shape[1]
    assert np.allclose(np.hypot(xy[:, 0] - CENTER[0], xy[:, 1] - CENTER[1]), 40.0, atol=1e-3)


def test_slice_at_z_skips_edges_touching_invalid_points():
    x, y, z = cylinder_grid()
    valid = np.ones(x.shape, bool)
    valid[:, :90] = False
    xy = slice_at_z(x, y, z, valid, z0=35.5)
    theta = np.arctan2(xy[:, 1] - CENTER[1], xy[:, 0] - CENTER[0]) % (2 * np.pi)
    assert theta.min() > np.pi - 0.1  # only the valid half of the cylinder remains


def test_slice_at_z_outside_the_surface_is_empty():
    x, y, z = cylinder_grid(z_range=(10.0, 60.0))
    assert slice_at_z(x, y, z, np.ones(x.shape, bool), z0=200.0).shape == (0, 2)


def test_umbilicus_center_interpolates_between_control_points(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": [
        {"x": 200.0, "y": 80.0, "z": 100.0},
        {"x": 100.0, "y": 50.0, "z": 0.0},
    ]}))
    assert np.allclose(umbilicus_center(path, z=25.0), (125.0, 57.5))
    assert np.allclose(umbilicus_center(path, z=12.5, coordinate_scale=0.5), (62.5, 28.75))


def test_radius_profile_gives_radius_per_angle_and_nan_where_uncovered():
    x, y, z = cylinder_grid(radius=40.0, arc=np.pi)  # half a turn only
    xy = slice_at_z(x, y, z, np.ones(x.shape, bool), z0=35.5)
    profile = radius_profile(xy, CENTER, n_theta=72)
    assert np.allclose(profile[2:34], 40.0, atol=1e-3)
    assert np.isnan(profile[40:70]).all()


def test_build_stack_orders_windings_from_inside_out_and_rescales(tmp_path):
    from scrollorder.io_real import build_stack

    dirs = []
    for name, radius in (("w_outer", 110.0), ("w_inner", 80.0)):
        x, y, z = cylinder_grid(center_xy=(240.0, 180.0), radius=radius, z_range=(20.0, 120.0))
        dirs.append(write_tifxyz(tmp_path / f"{name}.tifxyz", x, y, z))
    # the probability slice is at pyramid level 1: every coordinate is halved
    stack = build_stack(dirs, z=35.25, center_xy=CENTER, n_theta=90, xyz_scale=0.5)
    assert stack.shape == (2, 90)
    assert np.allclose(stack[0], 40.0, atol=0.05) and np.allclose(stack[1], 55.0, atol=0.05)
