import time
import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import scrollorder.mass as mass
from scrollorder.curves import Curve, center_from_curves, curve_at_z
from scrollorder.mass import (ROW_KEYS, ROW_KEYS_EXTENDED, SliceMeasure, air_map, band_mass, integrate_profiles,
                              measure_slice, sample_profiles)
from scrollorder.paths import arc_ids, trace_paths
from scrollorder.phantom import make_volume_phantom
from scrollorder.thresholds import MIN_INCLINATION, MIN_VALID_PROFILE_SHARE
from scrollorder.volume import SliceImage


# --- the tests of the brief (fixture, helper and four tests, verbatim) --------------------------

@pytest.fixture(scope="module")
def ph_measure():
    ph = make_volume_phantom(join_winding=None)
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    return ph, measure_slice(img, curves, center_from_curves(curves), m_max=4)

def rows(meas, m, status="ok"):
    return [r for r in meas.rows if r["m"] == m and r["status"] == status]

def test_air_map_is_the_air_level(ph_measure):
    ph, meas = ph_measure
    assert np.abs(meas.air - 20.0).max() < 2.0

def test_sheet_mass_is_conserved_in_the_pressed_zone(ph_measure):
    ph, meas = ph_measure
    near = lambda r: abs(((r["theta"] - ph["pressed_theta"] + np.pi) % (2 * np.pi)) - np.pi) < 0.3
    clean = [r["mass"] for r in rows(meas, 1) if r["clean"]]
    pressed = [r["mass"] for r in rows(meas, 1) if near(r) and 2 <= r["k"] <= 5]
    assert np.median(clean) == pytest.approx(720.0, rel=0.02)
    assert np.median(pressed) == pytest.approx(np.median(clean), rel=0.04)
    assert meas.sigma_ref0 == pytest.approx(720.0, rel=0.02)

def test_mass_counts_every_sheet(ph_measure):
    ph, meas = ph_measure
    for m in (1, 2, 3, 4):
        counts = [round(r["mass"] / meas.sigma_ref0) for r in rows(meas, m)]
        assert counts and all(c == m for c in counts)
        band = [round(r["band_mass"] / meas.sigma_ref0) for r in rows(meas, m)]
        assert np.mean([b == m for b in band]) > 0.95

def test_constant_grey_undercounts_in_the_pressed_zone():
    ph = make_volume_phantom(join_winding=None)
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    meas = measure_slice(img, curves, center_from_curves(curves), m_max=2, constant_grey=True)
    near = [r for r in meas.rows if r["m"] == 2 and r["status"] == "ok" and r["k"] == 3
            and abs(((r["theta"] - ph["pressed_theta"] + np.pi) % (2 * np.pi)) - np.pi) < 0.1]
    assert near and all(round(r["mass"] / meas.sigma_ref0) < 2 for r in near)


# --- helpers ----------------------------------------------------------------------------------

def phantom_slice(**kw):
    """Plane 6 of the phantom without join, as a level-0 SliceImage, and its 8 curves."""
    ph = make_volume_phantom(join_winding=None, **kw)
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    return ph, img, curves


def wrapped_angle(theta, centre):
    return abs(((theta - centre + np.pi) % (2 * np.pi)) - np.pi)


# --- the quadrature: trapezoid on [0, t] with the exact endpoint --------------------------------

def test_integral_on_a_sample_and_between_two_samples():
    rho = np.array([[0.0, 2.0, 4.0, 1.0, 3.0, 5.0]])            # samples at s = 0, 0.5, ... 2.5
    at = lambda t: integrate_profiles(rho, np.array([t]), step=0.5)[0]
    assert at(0.0) == 0.0
    assert at(1.0) == pytest.approx(0.5 * (0 + 2) / 2 + 0.5 * (2 + 4) / 2)                  # 2.0, on a sample
    assert at(1.2) == pytest.approx(2.0 + 0.2 * (4.0 + 2.8) / 2)                            # rho(1.2) = 2.8
    assert at(0.2) == pytest.approx(0.2 * (0.0 + 0.8) / 2)
    assert at(2.5) == pytest.approx(0.5 + 1.5 + 1.25 + 1.0 + 2.0)                           # the last sample
    assert np.isnan(at(2.6))                                                                # no sample after t


def test_integral_never_reads_a_sample_beyond_the_one_after_t():
    rho = np.array([[0.0, 2.0, 4.0, 1.0, np.nan, np.nan]])
    assert integrate_profiles(rho, np.array([1.5]))[0] == pytest.approx(0.5 + 1.5 + 1.25)  # on sample 3
    assert np.isnan(integrate_profiles(rho, np.array([1.6]))[0])                            # needs sample 4
    rho2 = np.array([[0.0, 2.0, np.nan, 1.0, 3.0, 5.0]])
    assert integrate_profiles(rho2, np.array([0.5]))[0] == pytest.approx(0.5)
    assert np.isnan(integrate_profiles(rho2, np.array([0.7]))[0])


def test_integral_is_exact_on_linear_profiles_and_takes_columns_of_endpoints():
    rng = np.random.default_rng(1)
    s = 0.5 * np.arange(60)
    a, b = rng.uniform(-50, 50, 40), rng.uniform(-5, 5, 40)
    rho = a[:, None] + b[:, None] * s[None, :]
    t = rng.uniform(0, 29.0, (40, 3))
    t[0, 0] = 12.0                                              # exactly on a sample
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = integrate_profiles(rho, t)
    np.testing.assert_allclose(got, a[:, None] * t + b[:, None] * t ** 2 / 2, rtol=1e-12, atol=1e-9)
    t[1, 2] = np.nan
    got = integrate_profiles(rho, t)
    assert np.isnan(got[1, 2]) and np.isfinite(np.delete(got.ravel(), 1 * 3 + 2)).all()


def test_integral_of_a_box_does_not_depend_on_where_the_endpoint_falls():
    # one sheet of 4 px on a void, sampled at 0.5 px: the trapezoid with the exact endpoint gives the same
    # mass wherever t falls in the void (a plain sum of the samples up to t would not)
    s = 0.5 * np.arange(40)
    rho = np.where((s >= 1.0) & (s <= 5.0), 180.0, 0.0)[None, :]
    got = integrate_profiles(np.repeat(rho, 5, axis=0), np.array([8.0, 8.1, 8.25, 9.0, 11.9]))
    np.testing.assert_allclose(got, 180.0 * 4.0 + 180.0 * 0.5, rtol=1e-12)    # the two ramps of 0.5 px add half a step


# --- profiles ---------------------------------------------------------------------------------

def test_profiles_are_bilinear_along_the_rays_in_window_coordinates():
    yy, xx = np.mgrid[0:40, 0:50]
    data = (2.0 * xx + 7.0 * yy + 1.0).astype(np.float32)      # different slopes: an x / y swap shows
    img = SliceImage(data, (5, 3), 2, 0, 0.0, 0.01)             # a non-zero origin is NOT subtracted again
    start = np.array([[10.25, 12.5], [10.25, 12.5], [3.0, 4.0]])
    nu = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    got = sample_profiles(img, start, nu, n_samples=5, step=0.5)
    s = 0.5 * np.arange(5)
    x = start[:, :1] + s * nu[:, :1]
    y = start[:, 1:] + s * nu[:, 1:]
    assert got.shape == (3, 5) and got.dtype == np.float32
    np.testing.assert_allclose(got, 2 * x + 7 * y + 1, rtol=1e-6)
    np.testing.assert_allclose(sample_profiles(img, start, nu, 3), got[:, :3])            # default step 0.5


def test_profiles_are_nan_outside_the_image_and_for_a_nan_ray():
    img = SliceImage(np.ones((20, 20), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    start = np.array([[18.0, 5.0], [np.nan, 5.0], [5.0, 5.0]])
    nu = np.array([[1.0, 0.0], [1.0, 0.0], [np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sample_profiles(img, start, nu, 4)
    np.testing.assert_array_equal(got[0], [1.0, 1.0, 1.0, np.nan])        # x = 19 is the last pixel centre
    assert np.isnan(got[1:]).all()
    assert sample_profiles(img, np.zeros((0, 2)), np.zeros((0, 2)), 4).shape == (0, 4)


# --- the air map ------------------------------------------------------------------------------

def void_profile(void_at, void_grey, n=40, sheet=200.0):
    """A profile of `n` samples at 0.5 px: `sheet` everywhere, `void_grey` on the 8 samples from `void_at` px."""
    g = np.full(n, sheet)
    i = int(round(void_at / 0.5))
    g[i:i + 8] = void_grey
    return g


def tiles_after(filled):
    return gaussian_filter(np.asarray(filled, float), sigma=1.0, mode="nearest")


def test_air_is_the_lowest_four_pixel_mean_inside_the_void():
    img = SliceImage(np.zeros((120, 120), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    g = void_profile(6.0, 50.0)
    g[0:2] = 0.0                 # before 1 px: not in the void
    g[6:8] = 10.0                # a dip of 1 px: narrower than void_min_px = 4 px
    g[29:37] = 5.0               # after t - 1 px = 14 px: beyond the next sheet's edge
    grey = np.tile(g, (6, 1))
    start = np.tile([5.0, 5.0], (6, 1))
    nu = np.tile([1.0, 0.0], (6, 1))
    air = air_map(img, grey, np.full(6, 15.0), start, nu, tile_px=40.0)
    assert air.shape == img.data.shape
    np.testing.assert_allclose(air, 50.0, rtol=1e-6)          # one tile with data: the same value everywhere


def test_air_keeps_the_25th_percentile_of_a_tile():
    img = SliceImage(np.zeros((120, 120), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    values = np.arange(10.0, 90.0, 10.0)
    grey = np.stack([void_profile(4.0, v) for v in values])
    air = air_map(img, grey, np.full(8, 15.0), np.tile([5.0, 5.0], (8, 1)), np.tile([0.0, 1.0], (8, 1)), 40.0)
    np.testing.assert_allclose(air, np.percentile(values, 25), rtol=1e-6)


def two_tile_profiles(n_sparse=0):
    """5 profiles of air 20 in tile (0, 0) and 5 of air 80 whose void lies in tile (0, 1) although they
    start in tile (0, 0); `n_sparse` profiles of air 100 in tile (2, 2)."""
    grey = [void_profile(2.0, 20.0)] * 5 + [void_profile(12.0, 80.0)] * 5 + [void_profile(2.0, 100.0)] * n_sparse
    start = [[5.0, 20.0]] * 5 + [[30.0, 20.0]] * 5 + [[100.0, 100.0]] * n_sparse
    n = len(grey)
    return np.array(grey), np.full(n, 18.0), np.array(start), np.tile([1.0, 0.0], (n, 1))


def test_air_values_are_placed_where_the_void_is_and_tiles_without_data_take_the_nearest():
    img = SliceImage(np.zeros((120, 120), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    air = air_map(img, *two_tile_profiles(), tile_px=40.0)
    # nearest tile with data, worked out by hand: column 0 -> tile (0, 0), columns 1 and 2 -> tile (0, 1)
    smooth = tiles_after([[20, 80, 80], [20, 80, 80], [20, 80, 80]])
    for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 0)]:
        assert air[40 * i + 20, 40 * j + 20] == pytest.approx(smooth[i, j], rel=1e-6)     # tile centres
    assert air[20, 40] == pytest.approx((smooth[0, 0] + smooth[0, 1]) / 2, rel=1e-6)      # bilinear between them
    assert air[0, 0] == pytest.approx(smooth[0, 0], rel=1e-6)                             # flat beyond the centres


def test_a_tile_with_fewer_than_five_values_takes_the_value_of_its_neighbour():
    img = SliceImage(np.zeros((120, 120), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    base = air_map(img, *two_tile_profiles(), tile_px=40.0)
    np.testing.assert_array_equal(air_map(img, *two_tile_profiles(n_sparse=4), tile_px=40.0), base)
    five = air_map(img, *two_tile_profiles(n_sparse=5), tile_px=40.0)
    assert five[100, 100] > base[100, 100] + 5


def test_air_map_refuses_to_guess():
    img = SliceImage(np.zeros((120, 120), np.float32), (0, 0), 0, 0, 0.0, 0.05)
    grey, t1, start, nu = two_tile_profiles()
    with pytest.raises(ValueError, match="air"):
        air_map(img, grey[:4], t1[:4], start[:4], nu[:4], 40.0)          # no tile with 5 values
    with pytest.raises(ValueError, match="air"):
        air_map(img, grey, np.full(len(grey), 5.0), start, nu, 40.0)    # [1, 4] px: no room for 4 px


# --- the band variant -------------------------------------------------------------------------

def test_band_mass_sums_the_pixel_centres_of_the_polygon_over_the_mean_edge():
    rho = np.full((50, 50), 3.0, np.float32)
    inner = np.stack([np.linspace(10.5, 20.5, 11), np.full(11, 10.5)], axis=1)
    outer = np.stack([np.linspace(10.5, 20.5, 6), np.full(6, 30.5)], axis=1)
    assert band_mass(rho, inner, outer) == pytest.approx(3.0 * 10 * 20 / 10.0)   # x 11..20, y 11..30
    wide = np.stack([np.linspace(5.5, 25.5, 6), np.full(6, 30.5)], axis=1)       # outer edge twice as long
    yy, xx = np.mgrid[0:50, 0:50]
    half = 5.0 + 5.0 * (yy - 10.5) / 20.0                                        # half width of the trapezoid
    inside = (yy > 10.5) & (yy < 30.5) & (np.abs(xx - 15.5) < half)
    assert band_mass(rho, inner, wide) == pytest.approx(3.0 * inside.sum() / 15.0)
    shifted = band_mass(rho, inner + [35.0, 0.0], outer + [35.0, 0.0])           # half outside the image
    assert shifted == pytest.approx(3.0 * 4 * 20 / 10.0)                         # x 46..49 only
    assert np.isnan(band_mass(rho, inner[:1], outer[:1]))                        # no edge length


def test_a_folded_band_is_a_discard_and_a_clean_one_is_not():
    rho = np.ones((60, 60), np.float32)
    inner = np.stack([np.linspace(10.5, 20.5, 11), np.full(11, 10.5)], axis=1)
    outer = np.stack([np.linspace(10.5, 20.5, 11), np.full(11, 30.5)], axis=1)
    assert band_mass(rho, inner, outer) == pytest.approx(200 / 10.0)              # convex, in order: finite
    assert np.isnan(band_mass(rho, inner, outer[::-1]))                           # arrivals reversed: a bow tie
    bent = outer.copy()
    bent[6] = [14.0, 30.5]                                                        # one arrival step goes back
    assert np.isnan(band_mass(rho, inner, bent))
    back = inner.copy()
    back[5] = [9.0, 10.5]                                                         # the start edge doubles back
    assert np.isnan(band_mass(rho, back, outer))
    ring = np.stack([15.5 + 5 * np.cos(np.linspace(0, 2 * np.pi, 9)), 15.5 + 5 * np.sin(np.linspace(0, 2 * np.pi, 9))], 1)
    assert np.isnan(band_mass(rho, ring, outer))                                  # first point = last point
    tilted = outer + np.stack([np.zeros(11), np.linspace(0.0, 3.0, 11)], axis=1)  # oblique but in order
    assert np.isfinite(band_mass(rho, inner, tilted))


def test_the_folded_bands_of_the_join_flank_are_discarded_not_miscounted():
    # the reviewer's case: on the default phantom the normals of curve 4 converge where it bends at the
    # flank of the join, the arrival points of arcs 27 and 45 at m = 3 fold back; the band gave 2 for 3
    ph = make_volume_phantom()
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    meas = measure_slice(img, curves, center_from_curves(curves), m_max=4)
    for arc in (27, 45):
        r = row_of(meas, 4, arc, 3)
        assert r["status"] == "ok" and round(r["mass"] / meas.sigma_ref0) == 3 and np.isnan(r["band_mass"])
    ok = [r for r in meas.rows if r["status"] == "ok"]
    finite = [r for r in ok if np.isfinite(r["band_mass"])]
    assert all(round(r["band_mass"] / meas.sigma_ref0) == round(r["mass"] / meas.sigma_ref0) for r in finite)
    assert len(ok) - len(finite) <= 10


def test_ok_guarantees_a_finite_mass_but_not_a_finite_band_mass():
    ph, img, curves = phantom_slice()
    centre = np.array(center_from_curves(curves))
    # a one-point arc: point 200 of curve 1 moved 12 px along its ring (> ARC_MM = 10 px from both neighbours)
    xy = curves[1].xy.copy()
    radius = np.hypot(*(xy[200] - centre))
    angle = np.arctan2(xy[200, 1] - centre[1], xy[200, 0] - centre[0]) + 12.0 / radius
    xy[200] = centre + radius * np.array([np.cos(angle), np.sin(angle)])
    curves[1] = Curve(curves[1].name, xy, curves[1].u_mm, curves[1].normal3d)
    lone = arc_ids(curves[1], img.px_mm)[200]
    assert (arc_ids(curves[1], img.px_mm) == lone).sum() == 1
    # a NaN pixel inside sheet 0 at theta = pi / 2 (middle of arc 4): the profiles through it are invalid,
    # the others are not
    data = img.data.copy()
    data[int(centre[1]) + 32, int(centre[0])] = np.nan
    holed = SliceImage(data, img.origin_xy, img.level, img.plane, img.z_l0, img.px_mm)
    meas = measure_slice(holed, curves, tuple(centre), m_max=2)
    for m in (1, 2):
        r = row_of(meas, 1, lone, m)
        assert r["status"] == "ok" and r["n_profiles"] == 1 and np.isfinite(r["mass"]) and np.isnan(r["band_mass"])
    pixel = [r for r in meas.rows if r["k"] == 0 and r["status"] == "ok" and np.isnan(r["band_mass"])]
    assert pixel and all(r["n_profiles"] > 1 and np.isfinite(r["mass"]) for r in pixel)
    # what the docstring of SliceMeasure states, on every "ok" row
    for r in meas.rows:
        if r["status"] == "ok":
            assert np.isfinite([r["mass"], r["spacing_px"], r["c"]]).all() and r["c"] >= MIN_INCLINATION


# --- the measure of the phantom ---------------------------------------------------------------

def test_rows_have_the_keys_and_plain_python_types(ph_measure):
    _, meas = ph_measure
    assert isinstance(meas, SliceMeasure)
    assert ROW_KEYS == ("k", "arc", "m", "theta", "u_mm", "c", "spacing_px", "mass", "band_mass", "n_profiles",
                        "status", "clean", "u_path")
    assert ROW_KEYS_EXTENDED == ROW_KEYS + ("motif",)
    types = {"k": int, "arc": int, "m": int, "theta": float, "u_mm": float, "c": float, "spacing_px": float,
             "mass": float, "band_mass": float, "n_profiles": int, "status": str, "clean": bool, "u_path": list,
             "motif": str}
    for r in meas.rows:
        assert tuple(r) == ROW_KEYS_EXTENDED
        assert all(type(r[key]) is kind for key, kind in types.items())
        assert len(r["u_path"]) == r["m"] and all(type(u) is float for u in r["u_path"])


def test_rows_exist_only_for_sheets_that_exist_and_arcs_that_exist(ph_measure):
    ph, meas = ph_measure
    n = len(ph["surfaces"])                                         # 8 curves
    assert {(r["k"], r["m"]) for r in meas.rows} == {(k, m) for k in range(n) for m in range(1, 5) if k + m <= n - 1}
    _, img, curves = phantom_slice()
    for k in range(n - 1):
        arc = meas.profiles[k]["arc"]                               # arc of the start point of each path
        expected = set(np.unique(arc[arc >= 0]).tolist())
        ids = arc_ids(curves[k], img.px_mm)
        assert expected == set(np.unique(ids[ids >= 0]).tolist())   # on the phantom every arc has paths
        assert (arc == -1).any()                                    # and the end of each curve has none
        for m in range(1, min(4, n - 1 - k) + 1):
            assert {r["arc"] for r in meas.rows if r["k"] == k and r["m"] == m} == expected


def test_the_phantom_has_only_ok_rows_and_whole_arcs(ph_measure):
    _, meas = ph_measure
    assert {r["status"] for r in meas.rows} == {"ok"}
    assert all(r["motif"] == "" for r in meas.rows)
    for r in meas.rows:
        assert np.isfinite([r["mass"], r["band_mass"], r["spacing_px"], r["c"]]).all()


def test_profiles_hold_what_t0_and_t1_need(ph_measure):
    ph, meas = ph_measure
    _, img, curves = phantom_slice()
    centre = center_from_curves(curves)
    assert sorted(meas.profiles) == list(range(len(curves) - 1))
    for k, prof in meas.profiles.items():
        assert set(prof) == {"grey", "t", "c", "arc", "start_xy", "nu"}
        p = trace_paths(curves, k, centre, 4, 5 * 12.0)              # same start points whatever max_len
        n = len(p.start_xy)
        assert prof["grey"].shape[0] == n and prof["t"].shape == prof["c"].shape == (n, 4)
        np.testing.assert_array_equal(prof["start_xy"], p.start_xy)
        np.testing.assert_array_equal(prof["nu"], p.nu)
        np.testing.assert_array_equal(prof["arc"], arc_ids(curves[k], img.px_mm)[p.point_index])
        ok = np.isfinite(prof["t"])
        np.testing.assert_allclose(prof["c"][ok], ((p.c_start[:, None] + p.c_end) / 2)[ok], rtol=1e-12)
        # raw grey, not air-subtracted: the sheets are at 200 and the voids at 20
        assert np.nanmax(prof["grey"]) == pytest.approx(200.0, abs=0.5)
        assert np.nanmin(prof["grey"]) == pytest.approx(20.0, abs=0.5)
        np.testing.assert_allclose(prof["grey"][:, :3], sample_profiles(img, p.start_xy, p.nu, 3), rtol=1e-6)


def test_clean_rows_sigma_ref0_and_s_clean_follow_their_definitions(ph_measure):
    _, meas = ph_measure
    m1 = rows(meas, 1)
    median = np.median([r["spacing_px"] for r in m1])
    assert all(r["clean"] == (r["spacing_px"] > median) for r in m1)
    assert not any(r["clean"] for r in meas.rows if r["m"] > 1)
    clean = [r for r in m1 if r["clean"]]
    assert 0 < len(clean) < len(m1)
    assert meas.sigma_ref0 == np.median([r["mass"] for r in clean])
    assert meas.s_clean == np.median([r["spacing_px"] for r in clean])
    assert meas.s_clean == pytest.approx(12.0, abs=0.2)


def test_theta_and_u_are_those_of_the_middle_of_the_arc(ph_measure):
    _, meas = ph_measure
    _, img, curves = phantom_slice()
    cx, cy = center_from_curves(curves)
    for r in meas.rows:
        curve = curves[r["k"]]
        idx = np.flatnonzero(arc_ids(curve, img.px_mm) == r["arc"])
        mid = idx[len(idx) // 2]
        x, y = curve.xy[mid]
        assert r["theta"] == pytest.approx(np.arctan2(y - cy, x - cx) % (2 * np.pi), abs=1e-12)
        assert r["u_mm"] == curve.u_mm[mid]
        assert abs(r["u_path"][0] - r["u_mm"]) < 0.05 * len(idx) / 2 + 1e-9     # median of the starts


def test_spacing_is_the_corrected_distance_per_sheet(ph_measure):
    _, meas = ph_measure
    free = [r for r in meas.rows if wrapped_angle(r["theta"], 0.0) < 0.3 and r["k"] <= 3]
    assert free and all(r["spacing_px"] == pytest.approx(12.0, abs=0.05) for r in free)
    assert all(r["c"] == pytest.approx(1.0, abs=1e-3) for r in free)


# --- rules on arcs: short, inclined, motif ------------------------------------------------------

def spoil(p, rows_, motif, col=0):
    """Make the paths `rows_` invalid from the column `col` on, as trace_paths does."""
    p.status[rows_, col:] = motif
    for a in (p.t, p.c_end, p.u_path):
        a[rows_, col:] = np.nan


@pytest.fixture(scope="module")
def spoiled():
    """The phantom measured with m_max = 2, the paths of some arcs of curve 0 made invalid by hand."""
    ph, img, curves = phantom_slice()
    ids = arc_ids(curves[0], img.px_mm)
    plan = {}
    real = trace_paths

    def edit(*args, **kwargs):
        p = real(*args, **kwargs)
        k, m_max = args[1], kwargs["m_max"]
        if k != 0 or m_max == 1:
            return p
        arc = ids[p.point_index]
        even = [a for a in range(arc.max() + 1) if (arc == a).sum() % 2 == 0]
        half, more, late, gone, tilt, edge = even[:6]
        rows_of = {a: np.flatnonzero(arc == a) for a in even[:6]}
        n = {a: len(r) for a, r in rows_of.items()}
        spoil(p, rows_of[half][: n[half] // 2], "self")                      # exactly half: not short
        r = rows_of[more][: n[more] // 2 + 1]
        spoil(p, r[:1], "hole")
        spoil(p, r[1:], "self")                                              # one more than half: short, "self"
        spoil(p, rows_of[late][: n[late] // 2 + 1], "order", col=1)         # m = 1 fine, m = 2 short
        spoil(p, rows_of[gone], "hole")                                      # nothing valid
        p.c_start[rows_of[tilt]] = 0.0
        p.c_end[rows_of[tilt], 0] = 0.0                                      # c = 0 at m = 1
        p.c_end[rows_of[tilt], 1] = 1.0                                      # c = 0.5 exactly at m = 2
        p.c_start[rows_of[edge]] = 0.4
        p.c_end[rows_of[edge], :] = 0.6                                      # c = 0.5 exactly
        plan.update(half=half, more=more, late=late, gone=gone, tilt=tilt, edge=edge, n=n)
        return p

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mass, "trace_paths", edit)
        meas = measure_slice(img, curves, center_from_curves(curves), m_max=2)
    return plan, meas


def row_of(meas, k, arc, m):
    (r,) = [r for r in meas.rows if r["k"] == k and r["arc"] == arc and r["m"] == m]
    return r


def test_half_the_profiles_valid_is_enough(spoiled):
    plan, meas = spoiled
    assert MIN_VALID_PROFILE_SHARE == 0.5
    for m in (1, 2):
        r = row_of(meas, 0, plan["half"], m)
        assert r["status"] == "ok" and r["n_profiles"] == plan["n"][plan["half"]] // 2 and r["motif"] == ""


def test_fewer_than_half_valid_is_short_with_the_most_frequent_motif(spoiled):
    plan, meas = spoiled
    for m in (1, 2):
        r = row_of(meas, 0, plan["more"], m)
        assert r["status"] == "short" and r["motif"] == "self"
        assert r["n_profiles"] == plan["n"][plan["more"]] - (plan["n"][plan["more"]] // 2 + 1)
        assert np.isnan([r["mass"], r["band_mass"], r["spacing_px"]]).all()
        assert not r["clean"]
    assert row_of(meas, 0, plan["late"], 1)["status"] == "ok"
    late = row_of(meas, 0, plan["late"], 2)
    assert late["status"] == "short" and late["motif"] == "order"
    gone = row_of(meas, 0, plan["gone"], 1)
    assert gone["status"] == "short" and gone["motif"] == "hole" and gone["n_profiles"] == 0
    assert np.isnan(gone["c"]) and all(np.isnan(gone["u_path"]))


def test_an_arc_below_the_minimum_inclination_is_inclined(spoiled):
    plan, meas = spoiled
    assert MIN_INCLINATION == 0.5
    r = row_of(meas, 0, plan["tilt"], 1)
    assert r["status"] == "inclined" and r["c"] == 0.0 and r["motif"] == ""
    assert np.isnan([r["mass"], r["band_mass"], r["spacing_px"]]).all()
    assert row_of(meas, 0, plan["tilt"], 2)["status"] == "ok"            # c = 0.5 is not below 0.5
    edge = row_of(meas, 0, plan["edge"], 1)
    assert edge["status"] == "ok" and edge["c"] == 0.5
    assert edge["mass"] == pytest.approx(720.0 / 2, rel=0.05)           # the mass carries the factor c


def test_the_other_arcs_are_untouched_by_the_spoiled_ones(spoiled):
    plan, meas = spoiled
    hit = {plan[key] for key in ("half", "more", "late", "gone", "tilt", "edge")}
    others = [r for r in meas.rows if not (r["k"] == 0 and r["arc"] in hit)]
    assert {r["status"] for r in others} == {"ok"}


def test_a_profile_whose_grey_is_missing_is_not_valid():
    ph, img, curves = phantom_slice()
    data = img.data.copy()
    data[:, :40] = np.nan                                  # x < 40: the rings of radius >= 90 on the left side
    holed = SliceImage(data, img.origin_xy, img.level, img.plane, img.z_l0, img.px_mm)
    meas = measure_slice(holed, curves, center_from_curves(curves), m_max=2)
    left = [r for r in meas.rows if r["k"] == 5 and wrapped_angle(r["theta"], np.pi) < 0.2]
    assert left and all(r["status"] == "short" and r["n_profiles"] == 0 and r["motif"] == "" for r in left)
    ok = [r for r in meas.rows if r["status"] == "ok"]
    assert ok and np.isfinite([[r["mass"], r["band_mass"]] for r in ok]).all()
    assert np.isfinite(meas.air).all()


# --- the two passes ---------------------------------------------------------------------------

def test_the_first_pass_sets_the_length_of_the_rays_of_the_full_pass(monkeypatch):
    _, img, curves = phantom_slice()
    calls = []
    real = trace_paths

    def spy(*args, **kwargs):
        p = real(*args, **kwargs)
        calls.append((args[1], kwargs["m_max"], kwargs["max_len"], p))
        return p

    monkeypatch.setattr(mass, "trace_paths", spy)
    meas = measure_slice(img, curves, center_from_curves(curves), m_max=3)
    n = len(curves) - 1
    assert [(k, m) for k, m, _, _ in calls] == [(k, 1) for k in range(n)] + [(k, 3) for k in range(n)]
    assert all(length == 1 / img.px_mm for _, m, length, _ in calls[:n])
    spacing = np.concatenate([p.t[p.status[:, 0] == "ok", 0] * (p.c_start + p.c_end[:, 0])[p.status[:, 0] == "ok"] / 2
                              for _, _, _, p in calls[:n]])
    assert all(length == pytest.approx(4 * np.median(spacing), rel=1e-12) for _, _, length, _ in calls[n:])
    assert meas.profiles[0]["grey"].shape[1] == int(np.floor(4 * np.median(spacing) / 0.5)) + 2


def test_a_first_pass_without_any_path_is_an_error():
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = lambda r: Curve(f"r{r}", np.stack([100 + r * np.cos(th), 100 + r * np.sin(th)], 1), th * 1.0,
                             np.stack([np.cos(th), np.sin(th), 0 * th], 1))
    img = SliceImage(np.zeros((200, 200), np.float32), (0, 0), 0, 0, 0.0, 0.05)     # 1 mm = 20 px
    with pytest.raises(ValueError, match="first pass"):
        measure_slice(img, [circle(30), circle(60)], (100.0, 100.0), m_max=1)


def test_measure_slice_refuses_what_makes_no_sense():
    _, img, curves = phantom_slice()
    with pytest.raises(ValueError, match="two curves"):
        measure_slice(img, curves[:1], center_from_curves(curves), m_max=1)
    with pytest.raises(ValueError, match="m_max"):
        measure_slice(img, curves, center_from_curves(curves), m_max=0)


def test_a_slice_without_any_clean_arc_is_an_error(monkeypatch):
    _, img, curves = phantom_slice()
    real = trace_paths

    def ruin(*args, **kwargs):
        p = real(*args, **kwargs)
        if kwargs["m_max"] > 1:
            spoil(p, np.arange(len(p.t)), "hole")
        return p

    monkeypatch.setattr(mass, "trace_paths", ruin)
    with pytest.raises(ValueError, match="clean"):
        measure_slice(img, curves, center_from_curves(curves), m_max=2)


# --- the negative control ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def constant():
    ph, img, curves = phantom_slice()
    return img, measure_slice(img, curves, center_from_curves(curves), m_max=2, constant_grey=True)


def test_constant_grey_is_one_on_the_same_samples_and_the_air_is_zero(constant):
    img, meas = constant
    assert (meas.air == 0).all() and meas.air.shape == img.data.shape
    for prof in meas.profiles.values():
        real = sample_profiles(img, prof["start_xy"], prof["nu"], prof["grey"].shape[1])
        np.testing.assert_array_equal(np.isnan(prof["grey"]), np.isnan(real))
        assert (prof["grey"][np.isfinite(prof["grey"])] == 1).all()


def test_constant_grey_mass_is_the_corrected_length_with_the_same_definitions(constant):
    _, meas = constant
    ok = [r for r in meas.rows if r["status"] == "ok"]
    assert ok and all(r["mass"] == pytest.approx(r["m"] * r["spacing_px"], rel=1e-9) for r in ok)
    clean = [r for r in ok if r["m"] == 1 and r["clean"]]
    assert meas.sigma_ref0 == np.median([r["mass"] for r in clean]) == pytest.approx(meas.s_clean, rel=1e-9)
    assert meas.s_clean == pytest.approx(12.0, abs=0.2)


# --- cost -------------------------------------------------------------------------------------

def _timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def test_profiles_of_a_real_sized_curve_are_not_sampled_or_integrated_in_a_python_loop():
    # a curve of a real slice: ~3000 start points, rays of (10 + 1) x ~27 px, i.e. ~600 samples each
    rng = np.random.default_rng(0)
    img = SliceImage(rng.random((1500, 1500)).astype(np.float32), (0, 0), 2, 0, 0.0, 0.0096)
    start = rng.uniform(300, 900, (3000, 2))
    angle = rng.uniform(0, 2 * np.pi, 3000)
    nu = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    t = rng.uniform(0, 299.0, (3000, 10))
    grey = sample_profiles(img, start, nu, 600)
    best_sample = min(_timed(lambda: sample_profiles(img, start, nu, 600)) for _ in range(3))
    best_integral = min(_timed(lambda: integrate_profiles(grey, t)) for _ in range(3))
    # generous bounds against a loop over the 1.8 million samples (~10 to 20 s), not a benchmark
    assert best_sample < 3.0, f"sample_profiles took {best_sample:.2f} s"          # ~0.05 s vectorised
    assert best_integral < 3.0, f"integrate_profiles took {best_integral:.2f} s"    # ~0.03 s vectorised
