import numpy as np
import pytest

from scrollorder.phantom import make_volume_phantom


def test_phantom_sheets_voids_and_pressed_zone():
    ph = make_volume_phantom()
    v, (cx, cy) = ph["volume"], ph["center_xy"]
    assert v.dtype == np.uint8 and v.shape[0] == 13
    r_in = ph["recto"](0, 0.0) + 2.0
    assert v[6, round(cy), round(cx + r_in)] == 200
    r_void = ph["recto"](1, 0.0) - 2.0
    assert v[6, round(cy), round(cx + r_void)] == 20
    th = ph["pressed_theta"]
    assert ph["recto"](4, th) - ph["recto"](3, th) == pytest.approx(ph["radial_thickness"](3, th), abs=0.05)
    assert ph["sheet_mass"] == 720.0


# --- what the later tasks (mass, joins) rely on ------------------------------------------------

def _ray_mass(ph, direction, r_lo, r_hi):
    """Sum of (grey - air) over the pixels of plane 6 on the axis ray `direction` (dx, dy), r in [r_lo, r_hi]."""
    cx, cy = (int(c) for c in ph["center_xy"])
    r = np.arange(int(r_lo), int(r_hi) + 1)
    grey = ph["volume"][6, cy + direction[1] * r, cx + direction[0] * r].astype(float)
    return float(np.sum(grey - 20.0)), grey


def test_one_sheet_has_the_known_mass_on_a_free_ray():
    ph = make_volume_phantom()
    mass, _ = _ray_mass(ph, (1, 0), 25, 38)               # sheet 0 alone, along +x
    assert mass == pytest.approx(720.0, rel=0.01)


def test_pressed_zone_is_one_contiguous_block_of_five_sheets():
    ph = make_volume_phantom()
    assert ph["pressed_theta"] == pytest.approx(np.pi / 2)
    lo = ph["recto"](2, np.pi / 2)
    hi = ph["recto"](6, np.pi / 2) + ph["radial_thickness"](6, np.pi / 2)
    assert hi - lo == pytest.approx(5 * 4.0, abs=0.05)    # windings 2 to 6: gaps 2, 3, 4, 5 are all closed
    mass, grey = _ray_mass(ph, (0, 1), int(lo) - 4, int(hi) + 4)       # +y is theta = pi / 2
    assert mass == pytest.approx(5 * 720.0, rel=0.01)
    inside = np.arange(int(lo) - 4, int(hi) + 5)
    assert np.all(grey[(inside > lo + 0.5) & (inside < hi - 0.5)] == 200)      # no air between the sheets


def test_the_join_doubles_winding_3_over_a_plateau_and_nowhere_else():
    ph = make_volume_phantom()
    jt = ph["join_theta"]
    assert jt == pytest.approx(3 * np.pi / 2)
    for d in (-0.9, -0.5, 0.0, 0.5, 0.9):                  # the plateau is join_width = 1.8 rad wide
        assert ph["radial_thickness"](3, jt + d) == pytest.approx(8.0, abs=0.05)
    for d in (-1.3, 1.3, np.pi):                           # past the 0.2 rad flanks
        assert ph["radial_thickness"](3, jt + d) == pytest.approx(4.0, abs=0.05)
    assert ph["radial_thickness"](2, jt) == pytest.approx(4.0, abs=0.05)
    assert ph["radial_thickness"](4, jt) == pytest.approx(4.0, abs=0.05)
    mass, _ = _ray_mass(ph, (0, -1), 62, 78)               # -y is theta = 3 pi / 2
    assert mass == pytest.approx(2 * 720.0, rel=0.01)
    plain = make_volume_phantom(join_winding=None)
    assert plain["radial_thickness"](3, jt) == pytest.approx(4.0, abs=0.05)


def test_perpendicular_thickness_stays_constant_where_the_ring_is_not_a_circle():
    ph = make_volume_phantom(join_winding=None)
    for k in (3, 4, 5):
        for theta in np.linspace(0.0, 2 * np.pi, 73):
            h = 1e-4
            slope = (ph["recto"](k, theta + h) - ph["recto"](k, theta - h)) / (2 * h)
            perpendicular = ph["radial_thickness"](k, theta) / np.sqrt(1 + (slope / ph["recto"](k, theta)) ** 2)
            assert perpendicular == pytest.approx(4.0, abs=0.02)


def test_rings_do_not_overlap_and_the_gaps_follow_the_pitch():
    ph = make_volume_phantom()
    theta = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    for k in range(7):
        gap = ph["recto"](k + 1, theta) - ph["recto"](k, theta) - ph["radial_thickness"](k, theta)
        assert gap.min() > -1e-9
        if k in (0, 1, 6):                                 # not pressed: pitch - thickness away from the join
            assert gap.max() == pytest.approx(8.0, abs=1e-6)
        else:
            assert gap.min() == pytest.approx(0.0, abs=1e-6)


def test_surfaces_lie_on_the_recto_and_are_the_same_in_every_plane():
    ph = make_volume_phantom()
    cx, cy = ph["center_xy"]
    assert len(ph["surfaces"]) == 8
    for k, (x, y, z, valid) in enumerate(ph["surfaces"]):
        assert x.shape == y.shape == z.shape == valid.shape == (13, 720) and valid.all()
        np.testing.assert_array_equal(z[:, 0], np.arange(13.0))
        np.testing.assert_array_equal(x, np.broadcast_to(x[0], x.shape))
        theta = 2 * np.pi * np.arange(720) / 720
        np.testing.assert_allclose(np.hypot(x[3] - cx, y[3] - cy), ph["recto"](k, theta), atol=1e-9)
        np.testing.assert_allclose(np.arctan2(y[3] - cy, x[3] - cx) % (2 * np.pi), theta, atol=1e-9)
    assert np.all(ph["volume"] == ph["volume"][0])         # identical planes (no noise)


def test_the_centre_is_in_the_middle_of_a_square_image_with_room_for_the_last_sheet():
    ph = make_volume_phantom()
    n_z, h, w = ph["volume"].shape
    assert h == w and h % 2 == 1
    assert ph["center_xy"] == ((w - 1) / 2, (h - 1) / 2)
    outer = max(ph["recto"](7, t) + ph["radial_thickness"](7, t) for t in np.linspace(0, 2 * np.pi, 361))
    assert (w - 1) / 2 > outer + 5
    assert ph["px_mm"] == 0.05


def test_options_noise_seed_and_grey_levels():
    a = make_volume_phantom(noise=5.0, seed=3)
    b = make_volume_phantom(noise=5.0, seed=3)
    c = make_volume_phantom(noise=5.0, seed=4)
    np.testing.assert_array_equal(a["volume"], b["volume"])
    assert not np.array_equal(a["volume"], c["volume"])
    assert a["volume"].dtype == np.uint8
    hi = make_volume_phantom(sheet_grey=255.0, air_grey=0.0, n_windings=3, n_z=5, n_cols=360, join_winding=None)
    assert hi["sheet_mass"] == 1020.0 and hi["volume"].shape[0] == 5 and len(hi["surfaces"]) == 3
    assert hi["volume"].max() == 255 and hi["volume"].min() == 0
    assert hi["surfaces"][0][0].shape == (5, 360)
