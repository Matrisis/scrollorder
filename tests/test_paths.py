import time
import warnings

import numpy as np
import pytest

from scrollorder.curves import Curve
from scrollorder.paths import Paths, arc_ids, ray_hits, trace_paths
from scrollorder.thresholds import ARC_MM


def circles(radii, n=720, gap=None):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = []
    for i, r in enumerate(radii):
        xy = np.stack([200 + r * np.cos(th), 200 + r * np.sin(th)], axis=1)
        if gap is not None and gap[0] == i:
            xy[gap[1]] = np.nan
        out.append(Curve(f"w{i}", xy, np.arange(n) * 0.1, np.stack([np.cos(th), np.sin(th), 0 * th], 1)))
    return out


def lines(ys, n=30):
    """Horizontal curves y = ys[i], one point per pixel for x = 0 .. n-1, normal3d pointing up (+y)."""
    x = np.arange(n, dtype=float)
    return [Curve(f"l{i}", np.stack([x, np.full(n, float(y))], axis=1), x * 0.1, np.tile([0.0, 1.0, 0.0], (n, 1)))
            for i, y in enumerate(ys)]


BELOW = (15.0, -1000.0)                # a centre far below the lines: "outward" is +y


def rows_by_point(p):
    """point_index -> row of the paths (the start points are the valid points of the curve)."""
    return {int(i): r for r, i in enumerate(p.point_index)}


# --- the four tests of the brief ---------------------------------------------------------------

def test_hits_are_the_gaps_between_circles():
    p = trace_paths(circles([50, 62, 74, 86]), k=0, center_xy=(200, 200), m_max=3, max_len=60)
    np.testing.assert_allclose(np.nanmedian(p.t, axis=0), [12, 24, 36], atol=0.05)
    assert (p.status == "ok").all()
    np.testing.assert_allclose(np.median(p.c_start), 1.0, atol=1e-3)

def test_hole_discards_longer_paths():
    p = trace_paths(circles([50, 62, 74, 86], gap=(2, slice(0, 30))), 0, (200, 200), 3, 60)
    hit = p.status[:, 1] == "hole"
    assert hit.sum() >= 25 and (p.status[hit, 2] == "hole").all() and (p.status[hit, 0] == "ok").all()

def test_self_crossing_is_discarded():
    cs = circles([50, 62, 74])
    cs[0].xy[10] = [200 + 70.0, 200 + 0.0]          # pli : la spire 0 repasse au-delà de la spire 1
    p = trace_paths(cs, 0, (200, 200), 2, 60)
    assert (p.status[:, 1] == "self").any() or (p.status[:, 0] == "self").any()

def test_arcs_have_the_requested_length():
    c = circles([100])[0]                                   # périmètre 628 px
    ids = arc_ids(c, px_mm=0.05, arc_mm=0.5)                # 10 px par arc
    sizes = np.bincount(ids[ids >= 0])
    assert len(sizes) == 62 and sizes.min() >= 9


# --- ray_hits ------------------------------------------------------------------------------------

def test_ray_hits_gives_the_distance_and_the_fractional_index():
    curve = lines([10.0], n=21)[0].xy
    t, frac = ray_hits(np.array([[3.25, 0.0], [7.0, 4.0]]), np.array([[0.0, 1.0], [0.0, 1.0]]), curve, 30.0)
    np.testing.assert_allclose(t, [10.0, 6.0])
    np.testing.assert_allclose(frac, [3.25, 7.0])
    assert t.shape == frac.shape == (2,)

def test_ray_hits_measures_a_distance_whatever_the_length_of_the_direction():
    curve = lines([10.0], n=21)[0].xy
    t, _ = ray_hits(np.array([[3.0, 0.0], [3.0, 0.0]]), np.array([[0.0, 3.0], [0.0, 0.25]]), curve, 30.0)
    np.testing.assert_allclose(t, [10.0, 10.0])

def test_ray_hits_is_nan_without_a_crossing_within_max_len():
    curve = lines([10.0], n=21)[0].xy
    starts = np.array([[3.0, 0.0], [3.0, 0.0], [-5.0, 0.0], [25.0, 0.0], [3.0, 20.0]])
    dirs = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    t, frac = ray_hits(starts, dirs, curve, 30.0)
    assert np.isnan(t[1:]).all() and np.isnan(frac[1:]).all()        # backwards, left of, right of, beyond the curve
    assert t[0] == pytest.approx(10.0)
    t, frac = ray_hits(starts[:1], dirs[:1], curve, 9.9)               # too far
    assert np.isnan(t).all() and np.isnan(frac).all()
    t, _ = ray_hits(starts[:1], dirs[:1], curve, 10.0)                 # exactly at max_len: kept
    assert t[0] == pytest.approx(10.0)

def test_ray_hits_ignores_crossings_closer_than_half_a_pixel():
    curve = lines([10.0], n=21)[0].xy
    starts = np.array([[3.25, 9.6], [3.25, 9.4], [3.25, 10.0], [3.25, 10.3]])
    t, _ = ray_hits(starts, np.tile([0.0, 1.0], (4, 1)), curve, 30.0)
    assert np.isnan(t[0]) and np.isnan(t[2])             # 0.4 px and 0 px
    assert t[1] == pytest.approx(0.6)
    assert np.isnan(t[3])                                # already past the curve

def test_ray_hits_takes_the_first_crossing():
    curve = np.array([[0.0, 10.0], [20.0, 10.0], [20.0, 14.0], [0.0, 14.0]])
    t, frac = ray_hits(np.array([[5.0, 0.0]]), np.array([[0.0, 1.0]]), curve, 30.0)
    np.testing.assert_allclose(t, [10.0])
    np.testing.assert_allclose(frac, [0.25])

def test_ray_hits_accepts_every_segment_of_a_curve_without_holes_ends_included():
    curve = lines([10.0], n=21)[0].xy
    xs = np.arange(21) - 0.5 + 1e-3
    xs = xs[(xs >= 0) & (xs <= 20)]
    t, _ = ray_hits(np.stack([xs, np.zeros_like(xs)], axis=1), np.tile([0.0, 1.0], (len(xs), 1)), curve, 30.0)
    assert np.isfinite(t).all()                          # the first and the last segment are real targets
    t, _ = ray_hits(np.array([[0.5, 0.0], [19.5, 0.0]]), np.tile([0.0, 1.0], (2, 1)), curve, 30.0)
    np.testing.assert_allclose(t, [10.0, 10.0])

def test_ray_hits_refuses_the_segments_next_to_a_hole():
    curve = lines([10.0], n=21)[0].xy.copy()
    curve[10] = np.nan                                   # (8, 9) and (11, 12) are the last segments before the hole
    xs = np.array([7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
    t, frac = ray_hits(np.stack([xs, np.zeros(6)], axis=1), np.tile([0.0, 1.0], (6, 1)), curve, 30.0)
    np.testing.assert_array_equal(np.isfinite(t), [True, False, False, False, False, True])
    np.testing.assert_allclose(frac[[0, 5]], [7.5, 12.5])
    np.testing.assert_allclose(t[[0, 5]], [10.0, 10.0])

def test_ray_hits_treats_a_hole_at_an_end_of_the_array_like_any_other():
    curve = lines([10.0], n=21)[0].xy.copy()
    curve[0] = np.nan                                    # (1, 2) borders the hole ; (0, 1) is not a segment
    curve[20] = np.nan                                   # (18, 19) borders the hole
    xs = np.array([0.5, 1.5, 2.5, 17.5, 18.5, 19.5])
    t, _ = ray_hits(np.stack([xs, np.zeros(6)], axis=1), np.tile([0.0, 1.0], (6, 1)), curve, 30.0)
    np.testing.assert_array_equal(np.isfinite(t), [False, False, True, True, False, False])

def test_ray_hits_goes_on_looking_when_the_first_segment_it_touches_is_refused():
    # y = 10 from x = 0 to 10 with a hole at point 5, then up to y = 20 and back along y = 20
    first = np.stack([np.arange(11.0), np.full(11, 10.0)], axis=1)
    first[5] = np.nan
    second = np.stack([np.arange(10.0, -1.0, -1.0), np.full(11, 20.0)], axis=1)
    curve = np.vstack([first, second])
    t, frac = ray_hits(np.array([[3.5, 0.0], [2.5, 0.0]]), np.tile([0.0, 1.0], (2, 1)), curve, 30.0)
    np.testing.assert_allclose(t, [20.0, 10.0])          # (3, 4) is refused: the ray reaches the return line
    np.testing.assert_allclose(frac, [11 + 6.5, 2.5])

def test_ray_hits_through_vertices_never_slips_between_two_segments():
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    unit = np.stack([np.cos(th), np.sin(th)], axis=1)
    t, frac = ray_hits(200 + 50 * unit, unit, 200 + 62 * unit, 60.0)    # each ray goes through a vertex
    assert np.isfinite(t).all()
    np.testing.assert_allclose(t, 12.0, atol=1e-3)
    np.testing.assert_allclose(frac, np.arange(720), atol=1e-6)

def test_ray_hits_through_a_vertex_survives_the_rounding_of_the_two_segments():
    # rays found by search that go through vertex i of this circle and that, with an exact 0 <= s <= 1 test,
    # slip between segments i - 1 and i (s = 1 + 1e-16 for the one, -1e-16 for the other)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    curve = 200 + 62 * np.stack([np.cos(th), np.sin(th)], axis=1)
    cases = [(296, [140.15514368528503, 229.50729817668133], [0.9082342941270699, 0.4184620257221711]),
             (314, [136.90284523975322, 229.48737191070325], [0.753231730774436, -0.6577552430460344]),
             (52, [259.23216745499184, 219.98864283547962], [-0.4383670730554318, 0.898796033180395])]
    for vertex, start, direction in cases:
        t, frac = ray_hits(np.array([start]), np.array([direction]), curve, 30.0)
        assert t[0] == pytest.approx(8.0, abs=1e-6) and frac[0] == pytest.approx(vertex, abs=1e-6), vertex

def test_ray_hits_degenerate_inputs_are_nan_without_any_warning():
    line = lines([10.0], n=21)[0].xy
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # parallel and collinear rays, a NaN start, a NaN direction, a zero direction
        t, frac = ray_hits(np.array([[-5.0, 10.0], [-5.0, 3.0], [np.nan, 0.0], [3.0, 0.0], [3.0, 0.0]]),
                           np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 1.0], [0.0, 0.0]]), line, 30.0)
        assert np.isnan(t).all() and np.isnan(frac).all()
        up, down = np.array([[3.0, 0.0]]), np.array([[0.0, 1.0]])
        for bad in (np.full((5, 2), np.nan), np.zeros((0, 2)), np.array([[3.0, 10.0]]), np.zeros((2, 2))):
            t, _ = ray_hits(up, down, bad, 30.0)                              # no usable segment
            assert np.isnan(t).all()
        t, frac = ray_hits(np.zeros((0, 2)), np.zeros((0, 2)), line, 30.0)    # no ray
        assert t.shape == frac.shape == (0,)
        t, _ = ray_hits(up, down, np.array([[0.0, 10.0], [5.0, 10.0], [5.0, 10.0], [10.0, 10.0]]), 30.0)
        assert t[0] == pytest.approx(10.0)                                    # a zero-length segment is no target

def test_ray_hits_on_ten_thousand_rays_is_not_a_python_loop():
    n = 10_000
    dtheta = 2 * np.pi / (n - 1)
    th = np.linspace(0, 2 * np.pi, n)                      # a closed curve of 10 000 points, 0.64 px apart
    curve = 1012.0 * np.stack([np.cos(th), np.sin(th)], axis=1)
    angle = np.linspace(0.37 * dtheta, 2 * np.pi - 0.63 * dtheta, n)       # 10 000 rays between the vertices
    rays = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    t0 = time.perf_counter()
    t, frac = ray_hits(1000.0 * rays, rays, curve, 60.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"{elapsed:.2f} s for 10 000 rays"
    np.testing.assert_allclose(t, 12.0, atol=1e-3)
    np.testing.assert_allclose(frac, angle / dtheta, atol=1e-4)


# --- trace_paths -------------------------------------------------------------------------------

def test_paths_start_at_the_valid_points_of_the_curve():
    p = trace_paths(circles([50, 62, 74, 86], gap=(0, slice(0, 30))), 0, (200, 200), 3, 60)
    assert p.k == 0
    np.testing.assert_array_equal(p.point_index, np.arange(30, 720))
    n_rays = len(p.point_index)
    cs = circles([50, 62, 74, 86])
    np.testing.assert_allclose(p.start_xy, cs[0].xy[30:])
    assert p.start_xy.shape == p.nu.shape == (n_rays, 2) and p.c_start.shape == (n_rays,)
    assert p.t.shape == p.c_end.shape == p.u_path.shape == p.status.shape == (n_rays, 3)
    np.testing.assert_allclose(np.hypot(p.nu[:, 0], p.nu[:, 1]), 1.0)
    assert (np.sum(p.nu * (p.start_xy - 200), axis=1) > 0).all()               # outward
    assert (p.status == "ok").all()

def test_status_is_a_unicode_array_of_8_characters():
    p = trace_paths(circles([50, 62]), 0, (200, 200), 1, 60)
    assert p.status.dtype == np.dtype("<U8") and p.status.shape == (720, 1)

def test_a_curve_without_start_point_gives_empty_paths():
    cs = circles([50, 62, 74])
    cs[0].xy[:] = np.nan
    p = trace_paths(cs, 0, (200, 200), 2, 60)
    assert p.start_xy.shape == p.nu.shape == (0, 2) and p.point_index.shape == p.c_start.shape == (0,)
    assert p.t.shape == p.c_end.shape == p.u_path.shape == p.status.shape == (0, 2)
    assert p.status.dtype == np.dtype("<U8")

def test_a_point_without_normal3d_is_not_a_start_point():
    cs = circles([50, 62, 74])
    cs[0].normal3d[5] = np.nan
    p = trace_paths(cs, 0, (200, 200), 2, 60)
    assert 5 not in p.point_index and len(p.point_index) == 719

def test_u_path_c_end_and_t_on_the_circles():
    p = trace_paths(circles([50, 62, 74, 86]), 0, (200, 200), 3, 60)
    inner = (p.point_index >= 1) & (p.point_index <= 718)                         # the ends have a one-sided normal
    # radial rays go through the vertex of the same index on every circle: u_mm = 0.1 * index everywhere
    expected_u = 0.1 * p.point_index[:, None] * np.ones((1, 3))
    np.testing.assert_allclose(p.u_path[inner], expected_u[inner], atol=1e-6)
    np.testing.assert_allclose(p.c_end[inner], 1.0, atol=1e-9)
    np.testing.assert_allclose(p.c_start[inner], 1.0, atol=1e-9)
    np.testing.assert_allclose(p.t[inner], np.tile([12.0, 24.0, 36.0], (inner.sum(), 1)), atol=0.01)

def test_the_inclination_is_the_absolute_dot_product_of_normal3d_and_nu():
    angle = 2 * np.pi / 3                                  # cos = -0.5: the sign of normal3d does not matter
    cs = []
    for c in circles([50, 62, 74]):
        nx, ny = c.normal3d[:, 0], c.normal3d[:, 1]
        tilted = np.stack([np.cos(angle) * nx - np.sin(angle) * ny, np.sin(angle) * nx + np.cos(angle) * ny,
                           np.zeros_like(nx)], axis=1)
        cs.append(Curve(c.name, c.xy, c.u_mm, tilted))
    p = trace_paths(cs, 0, (200, 200), 2, 60)
    inner = (p.point_index >= 1) & (p.point_index <= 718)
    np.testing.assert_allclose(p.c_start[inner], 0.5, atol=1e-9)
    np.testing.assert_allclose(p.c_end[inner], 0.5, atol=1e-9)

def test_u_mm_and_normal3d_are_interpolated_at_the_hit_and_the_normal_renormalised():
    cs = lines([0.0, 12.0], n=12)
    x = np.arange(12.0)
    phi = 0.4 * x                                          # the normal of curve 1 turns by 0.4 rad per point
    cs[1] = Curve("l1", np.stack([x + 0.5, np.full(12, 12.0)], axis=1), x * 0.1,
                  np.stack([np.sin(phi), np.cos(phi), 0 * phi], axis=1))
    p = trace_paths(cs, 0, BELOW, 1, 30.0)
    rows = rows_by_point(p)
    hit_rows = [rows[i] for i in range(1, 12)]                     # point i is half-way between points i-1 and i
    assert p.status[rows[0], 0] == "hole"                          # x = 0 is left of the first point of curve 1
    np.testing.assert_allclose(p.t[hit_rows, 0], 12.0)
    np.testing.assert_allclose(p.u_path[hit_rows, 0], 0.1 * np.arange(1, 12))        # the start point of curve 0
    np.testing.assert_allclose(p.c_end[hit_rows, 0], np.abs(np.cos(0.4 * (np.arange(1, 12) - 0.5))), atol=1e-12)
    # u_path[:, 1] would be the u_mm of the hit: check it on a two-sheet path
    cs.append(lines([24.0], n=12)[0])
    q = trace_paths(cs, 0, BELOW, 2, 40.0)
    qrows = [rows_by_point(q)[i] for i in range(1, 12)]
    np.testing.assert_allclose(q.u_path[qrows, 1], 0.1 * (np.arange(1, 12) - 0.5))   # enters sheet 1 between 2 points

def test_a_hole_in_a_curve_costs_exactly_the_rays_that_touch_its_neighbourhood():
    cs = lines([0.0, 12.0, 26.0, 38.0])
    cs[2].xy[10:15] = np.nan
    p = trace_paths(cs, 0, BELOW, 3, 60.0)
    rows = rows_by_point(p)
    holes = [i for i in range(30) if p.status[rows[i], 1] == "hole"]
    assert holes == list(range(9, 16))                    # the hole, and the segment on each side of it
    for i in range(30):
        expected = ["ok", "hole", "hole"] if i in holes else ["ok", "ok", "ok"]
        assert list(p.status[rows[i]]) == expected

def test_a_curve_that_is_not_in_the_list_is_a_hole():
    cs = circles([50, 62, 74])
    p = trace_paths(cs, 0, (200, 200), 4, 60)
    assert (p.status[:, :2] == "ok").all() and (p.status[:, 2:] == "hole").all()
    assert np.isnan(p.t[:, 2:]).all()
    q = trace_paths(cs, 2, (200, 200), 2, 60)             # nothing beyond the last curve
    assert (q.status == "hole").all() and q.t.shape == (720, 2)

def test_a_self_crossing_flags_the_columns_it_precedes():
    cs = lines([0.0, 12.0, 26.0, 38.0])
    cs[0].xy[20] = [14.0, 40.0]                            # a spike: the curve comes back over the points 14 to 19
    p = trace_paths(cs, 0, BELOW, 3, 60.0)
    rows = rows_by_point(p)
    expected = {                                           # t_self(x) = 8 * (19 - x) ; t = 12, 26, 38
        12: ["ok", "ok", "ok"],
        13: ["ok", "ok", "ok"],
        14: ["ok", "ok", "ok"],                            # through the tip: t_self = 40 > 38
        15: ["ok", "ok", "self"],                          # 32
        16: ["ok", "self", "self"],                        # 24
        17: ["ok", "self", "self"],                        # 16
        18: ["self", "self", "self"],                      # 8
    }
    for i, status in expected.items():
        assert list(p.status[rows[i]]) == status, i

def test_the_first_invalid_column_gives_the_reason_of_all_the_following_ones():
    cs = lines([0.0, 12.0, 26.0, 38.0])
    cs[0].xy[20] = [14.0, 40.0]
    cs[2].xy[14:18] = np.nan                               # the rays 13 to 18 touch this hole in column 1
    p = trace_paths(cs, 0, BELOW, 3, 60.0)
    rows = rows_by_point(p)
    assert list(p.status[rows[18]]) == ["self", "self", "self"]    # self in column 0, a hole in column 1
    assert list(p.status[rows[16]]) == ["ok", "hole", "hole"]      # a hole where self would apply
    assert list(p.status[rows[15]]) == ["ok", "hole", "hole"]
    assert list(p.status[rows[8]]) == ["ok", "ok", "ok"]

def test_hits_that_do_not_increase_are_an_order_error():
    p = trace_paths(lines([0.0, 24.0, 12.0, 36.0]), 0, BELOW, 3, 60.0)    # sheet 2 is met before sheet 1
    assert (p.status[:, 0] == "ok").all()
    assert (p.status[:, 1] == "order").all() and (p.status[:, 2] == "order").all()   # column 2 is fine by itself

def test_two_curves_crossed_at_the_same_distance_are_not_strictly_increasing():
    p = trace_paths(lines([0.0, 12.0, 12.0]), 0, BELOW, 2, 60.0)
    assert (p.status[:, 0] == "ok").all() and (p.status[:, 1] == "order").all()

def test_values_of_an_invalid_column_are_nan():
    p = trace_paths(lines([0.0, 24.0, 12.0, 36.0]), 0, BELOW, 3, 60.0)
    bad = p.status != "ok"
    assert bad.any()
    for field in (p.t, p.c_end, p.u_path):
        assert np.isnan(field[bad]).all() and np.isfinite(field[~bad]).all()
    assert np.isfinite(p.c_start).all()

def test_paths_are_traced_from_any_curve_of_the_list():
    p = trace_paths(circles([50, 62, 74, 86]), 1, (200, 200), 2, 60)
    assert p.k == 1
    np.testing.assert_allclose(np.nanmedian(p.t, axis=0), [12, 24], atol=0.05)
    assert (p.status == "ok").all()

def test_paths_without_point_index_start_at_the_first_points():
    p = Paths(0, np.zeros((3, 2)), np.ones((3, 2)), np.ones(3), np.ones((3, 1)), np.ones((3, 1)), np.ones((3, 1)),
              np.full((3, 1), "ok", dtype="<U8"))
    np.testing.assert_array_equal(p.point_index, [0, 1, 2])

def test_trace_paths_refuses_what_makes_no_sense():
    cs = circles([50, 62])
    for k in (-1, 2):
        with pytest.raises(ValueError, match="k"):
            trace_paths(cs, k, (200, 200), 1, 60)
    for m_max in (0, -1):
        with pytest.raises(ValueError, match="m_max"):
            trace_paths(cs, 0, (200, 200), m_max, 60)
    with pytest.raises(ValueError, match="max_len"):
        trace_paths(cs, 0, (200, 200), 1, 0.0)

def test_trace_paths_emits_no_warning_on_curves_full_of_holes():
    cs = circles([50, 62, 74], gap=(1, slice(100, 300)))
    cs[0].xy[400:500] = np.nan
    cs[2].normal3d[600:610] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = trace_paths(cs, 0, (200, 200), 3, 60)
    assert (p.status == "ok").any() and (p.status == "hole").any()


# --- arc_ids -----------------------------------------------------------------------------------

def test_arcs_count_is_consecutive_and_each_arc_is_contiguous():
    c = circles([100])[0]
    ids = arc_ids(c, px_mm=0.05, arc_mm=0.5)
    assert ids.shape == (720,) and np.issubdtype(ids.dtype, np.integer)
    sizes = np.bincount(ids[ids >= 0])
    assert (sizes >= 9).all() and (sizes <= 12).all()
    kept = ids[ids >= 0]
    assert (np.diff(kept) >= 0).all() and set(kept) == set(range(62))
    assert (ids[kept.size:] == -1).all()                   # the remainder of the circle (0.74 arc) closes the curve
    assert arc_ids(c, 0.05).tolist() == arc_ids(c, 0.05, ARC_MM).tolist()

def test_arcs_are_measured_in_mm_along_the_curve():
    line = lines([0.0], n=12)[0]                           # 1 px per point, 11 px long
    ids = arc_ids(line, px_mm=0.1, arc_mm=0.5)             # 5 px per arc
    assert ids.tolist() == [0] * 5 + [1] * 5 + [-1] * 2
    ids = arc_ids(line, px_mm=0.05, arc_mm=0.5)            # 10 px per arc: one arc, the last two points are left over
    assert ids.tolist() == [0] * 10 + [-1] * 2

def test_arcs_restart_after_each_hole_and_the_numbers_go_on():
    line = lines([0.0], n=25)[0]
    line.xy[12] = np.nan
    ids = arc_ids(line, px_mm=0.05, arc_mm=0.5)            # 10 px per arc
    # run 0..11 is 11 px long: one arc (points 0..9) and a remainder ; run 13..24 : one arc (13..22) and a remainder
    assert ids.tolist() == [0] * 10 + [-1] * 3 + [1] * 10 + [-1] * 2

def test_a_run_shorter_than_an_arc_gives_nothing():
    line = lines([0.0], n=8)[0]
    line.xy[3] = np.nan
    assert arc_ids(line, px_mm=0.05, arc_mm=0.5).tolist() == [-1] * 8
    assert arc_ids(Curve("c", np.full((6, 2), np.nan), np.arange(6.0), np.full((6, 3), np.nan)), 0.05).tolist() == [-1] * 6
    one = Curve("c", np.array([[3.0, 4.0]]), np.zeros(1), np.zeros((1, 3)))
    assert arc_ids(one, 0.05).tolist() == [-1]
    empty = Curve("c", np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3)))
    assert arc_ids(empty, 0.05).shape == (0,)

def test_arc_numbers_do_not_skip_a_run_that_has_no_arc():
    line = lines([0.0], n=30)[0]
    line.xy[[10, 14]] = np.nan                             # runs of 10, 3 and 15 points: 9 px, 2 px and 14 px long
    ids = arc_ids(line, px_mm=0.05, arc_mm=0.5)            # 10 px per arc: only the last run has an arc
    assert ids.tolist() == [-1] * 15 + [0] * 10 + [-1] * 5

def test_arc_ids_refuses_non_positive_lengths():
    c = circles([100])[0]
    for px_mm, arc_mm in ((0.0, 0.5), (-0.05, 0.5), (0.05, 0.0), (0.05, -0.5), (np.nan, 0.5), (0.05, np.inf)):
        with pytest.raises(ValueError):
            arc_ids(c, px_mm, arc_mm)
