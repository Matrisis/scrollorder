import dataclasses

import numpy as np
import pytest

from scrollorder.curves import Curve, center_from_curves, curve_at_z
from scrollorder.joins import JoinRun, confirm_joins, expected_layers, find_join_runs, in_join, sigma_ref_final
from scrollorder.mass import measure_slice
from scrollorder.phantom import make_volume_phantom
from scrollorder.thresholds import ARC_MM, JOIN_MASS, JOIN_MATCH_MM, JOIN_MIN_MM, JOIN_NEIGHBOUR_NORMAL
from scrollorder.volume import SliceImage


# --- the tests of the brief (fixture and four tests, verbatim) ----------------------------------

@pytest.fixture(scope="module")
def joined():
    ph = make_volume_phantom()
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    meas = measure_slice(img, curves, center_from_curves(curves), m_max=4)
    m1 = [r for r in meas.rows if r["m"] == 1]
    return ph, meas, find_join_runs(m1, meas.sigma_ref0)

def test_the_join_is_found_on_its_winding_only(joined):
    ph, meas, runs = joined
    assert [r.k for r in runs] == [3]
    mid = (runs[0].theta_lo + runs[0].theta_hi) / 2
    assert abs(((mid - ph["join_theta"] + np.pi) % (2 * np.pi)) - np.pi) < 0.2

def test_confirmation_needs_both_companions(joined):
    _, _, runs = joined
    assert confirm_joins(runs, [runs, runs]) == runs
    assert confirm_joins(runs, [runs, []]) == []

def test_heavy_neighbour_vetoes_a_join(joined):
    ph, meas, _ = joined
    m1 = [dict(r) for r in meas.rows if r["m"] == 1]
    for r in m1:
        if r["k"] == 4 and abs(((r["theta"] - ph["join_theta"] + np.pi) % (2 * np.pi)) - np.pi) < 0.9:
            r["mass"] *= 2
    assert all(run.k != 3 for run in find_join_runs(m1, meas.sigma_ref0))

def test_expected_layers_and_counts_through_the_join(joined):
    ph, meas, runs = joined
    crossing = [r for r in meas.rows if r["status"] == "ok" and r["k"] <= 3 < r["k"] + r["m"]
                and in_join(3, r["u_path"][3 - r["k"]], runs)]
    sref = sigma_ref_final([r for r in meas.rows if r["m"] == 1], runs)
    assert sref == pytest.approx(720.0, rel=0.02)
    assert crossing and all(round(r["mass"] / sref) == expected_layers(r["k"], r["m"], r["u_path"], runs) == r["m"] + 1
                            for r in crossing)


# --- the phantom: what the detector finds, and what it must not find ----------------------------

def test_the_run_covers_the_plateau_and_is_long_enough(joined):
    ph, meas, runs = joined
    (run,) = runs
    assert isinstance(run, JoinRun)
    assert run.u_hi - run.u_lo >= JOIN_MIN_MM
    plateau = 1.8 / 2                                           # join_width / 2, the cosine flanks are outside
    for theta in (run.theta_lo, run.theta_hi):                  # the arcs of the run sit on the plateau or its flank
        assert abs(((theta - ph["join_theta"] + np.pi) % (2 * np.pi)) - np.pi) < plateau + 0.2
    # the window of the run in u is the grid position of its angles: u = column * 0.05 mm, 720 columns a turn
    u_per_rad = 720 * 0.05 / (2 * np.pi)
    assert run.u_lo / u_per_rad == pytest.approx(run.theta_lo, abs=ARC_MM / 2 / u_per_rad + 0.15)
    assert run.u_hi / u_per_rad == pytest.approx(run.theta_hi, abs=ARC_MM / 2 / u_per_rad + 0.15)

def test_no_join_on_a_phantom_without_join_even_with_the_pressed_zone():
    ph = make_volume_phantom(join_winding=None)
    img = SliceImage(ph["volume"][6].astype(np.float32), (0, 0), 0, 6, 6.0, ph["px_mm"])
    curves = [Curve(f"w{k}", *curve_at_z(*s, 6.0, 0.05)) for k, s in enumerate(ph["surfaces"])]
    meas = measure_slice(img, curves, center_from_curves(curves), m_max=2)
    m1 = [r for r in meas.rows if r["m"] == 1]
    assert find_join_runs(m1, meas.sigma_ref0) == []
    assert sigma_ref_final(m1, []) == pytest.approx(meas.sigma_ref0)   # no join: the final reference is the first

def test_the_final_reference_leaves_the_join_out(joined):
    _, meas, runs = joined
    m1 = [r for r in meas.rows if r["m"] == 1]
    sref = sigma_ref_final(m1, runs)
    inside = [r for r in m1 if r["status"] == "ok" and in_join(r["k"], r["u_mm"], runs)]
    assert inside and all(r["mass"] / sref > 1.5 for r in inside)
    assert sref == pytest.approx(meas.sigma_ref0, rel=0.02)


# --- synthetic rows ----------------------------------------------------------------------------

SIGMA = 8.0                  # a power of two: mass / SIGMA is exact, so a bound is hit exactly
TWO_PI = 2 * np.pi

def row(k, arc, ratio, theta=None, u=None, status="ok", clean=True, m=1):
    """A row of a slice as `measure_slice` gives it: 0.25 rad and 0.5 mm per arc unless told otherwise."""
    theta = 0.25 * arc if theta is None else theta
    u = 0.5 * arc if u is None else u
    mass = ratio * SIGMA
    return {"k": k, "arc": arc, "m": m, "theta": float(theta) % TWO_PI, "u_mm": float(u), "c": 1.0,
            "spacing_px": 10.0 if status in ("ok", "band_nan") else float("nan"),
            "mass": mass if status in ("ok", "band_nan") else float("nan"),
            "band_mass": mass if status == "ok" else float("nan"), "n_profiles": 5, "status": status,
            "clean": clean, "u_path": [float(u)] * m, "motif": ""}

def curve_rows(k, arcs, ratio, **kw):
    """Rows of the arcs `arcs` of curve k, all at `ratio` sheets of mass (or `ratio(arc)` if callable)."""
    return [row(k, a, ratio(a) if callable(ratio) else ratio, **kw) for a in arcs]

def runs_of(rows, **kw):
    return find_join_runs(rows, SIGMA, **kw)

def scene(heavy=range(0, 12), below=1.0, above=1.0, arcs=range(0, 12)):
    """A join of 12 arcs (6 mm) on curve 3, neighbours on curves 2 and 4 (None: no row on that curve)."""
    rows = curve_rows(3, heavy, 2.0)
    if below is not None:
        rows += curve_rows(2, arcs, below)
    if above is not None:
        rows += curve_rows(4, arcs, above)
    return rows


# --- the run: arcs, gaps, length, mass range ---------------------------------------------------

def test_a_run_has_the_ruled_extent_and_angles():
    run, = runs_of(scene())
    assert run == JoinRun(k=3, u_lo=-ARC_MM / 2, u_hi=0.5 * 11 + ARC_MM / 2, theta_lo=0.0, theta_hi=0.25 * 11)
    assert isinstance(run.k, int)

def test_join_run_is_a_frozen_value():
    run, = runs_of(scene())
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.k = 4
    assert run in [dataclasses.replace(run)]

def test_rows_are_ordered_by_arc_not_by_position_in_the_list():
    rows = scene()
    rng = np.random.default_rng(1)
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    assert runs_of(shuffled) == runs_of(rows)

def test_a_gap_of_one_missing_arc_does_not_break_the_run():
    rows = curve_rows(3, [*range(0, 12), *range(13, 25)], 2.0) + curve_rows(2, range(0, 25), 1.0)
    run, = runs_of(rows)
    assert (run.u_lo, run.u_hi) == (-ARC_MM / 2, 0.5 * 24 + ARC_MM / 2)

def test_a_gap_of_two_missing_arcs_breaks_the_run():
    rows = curve_rows(3, [*range(0, 12), *range(14, 26)], 2.0) + curve_rows(2, range(0, 26), 1.0)
    first, second = runs_of(rows)
    assert (first.u_lo, first.u_hi) == (-ARC_MM / 2, 0.5 * 11 + ARC_MM / 2)
    assert (second.u_lo, second.u_hi) == (0.5 * 14 - ARC_MM / 2, 0.5 * 25 + ARC_MM / 2)

@pytest.mark.parametrize("middle", ["absent", "normal", "short"])
def test_a_gap_is_one_arc_whatever_the_arc_in_the_gap_is(middle):
    rows = curve_rows(3, [*range(0, 12), *range(13, 25)], 2.0) + curve_rows(2, range(0, 25), 1.0)
    if middle == "normal":
        rows.append(row(3, 12, 1.0))
    elif middle == "short":
        rows.append(row(3, 12, 2.0, status="short"))
    run, = runs_of(rows)
    assert run.u_hi == 0.5 * 24 + ARC_MM / 2

def test_a_gap_of_two_arcs_counts_present_arcs_that_do_not_qualify():
    rows = curve_rows(3, [*range(0, 12), *range(14, 26)], 2.0) + curve_rows(2, range(0, 26), 1.0)
    rows += [row(3, 12, 1.0), row(3, 13, 3.0)]                  # two arcs between the halves, none qualifies
    assert len(runs_of(rows)) == 2

def test_the_run_must_reach_the_minimum_length_and_five_mm_is_enough():
    ten = scene(heavy=range(0, 10))                             # 10 arcs of 0.5 mm: exactly 5.0 mm
    nine = scene(heavy=range(0, 9))                             # 4.5 mm
    run, = runs_of(ten)
    assert run.u_hi - run.u_lo == JOIN_MIN_MM == 5.0
    assert runs_of(nine) == []

def test_the_minimum_length_and_the_arc_length_are_parameters():
    nine = scene(heavy=range(0, 9))
    assert runs_of(nine, min_mm=4.5) != []
    assert runs_of(nine, min_mm=4.5 + 1e-9) == []
    assert runs_of(nine, arc_mm=1.0) != []                      # 4.0 mm between the centres + 1.0 mm: 5.0 mm

def test_two_short_pieces_are_not_a_join_even_if_together_they_are_long():
    rows = curve_rows(3, [*range(0, 7), *range(20, 27)], 2.0) + curve_rows(2, range(0, 27), 1.0)
    assert runs_of(rows) == []

def test_the_mass_range_bounds_are_inclusive():
    lo, hi = JOIN_MASS
    assert runs_of(scene(heavy=range(0, 12)), mass_range=(2.0, 2.0)) != []
    for ratio in (lo, hi):
        rows = curve_rows(3, range(0, 12), ratio) + curve_rows(2, range(0, 12), 1.0)
        assert len(runs_of(rows)) == 1
    below = curve_rows(3, range(0, 12), np.nextafter(lo, 0.0)) + curve_rows(2, range(0, 12), 1.0)
    above = curve_rows(3, range(0, 12), np.nextafter(hi, 9.0)) + curve_rows(2, range(0, 12), 1.0)
    assert runs_of(below) == [] and runs_of(above) == []

def test_a_run_may_mix_arcs_across_the_whole_mass_range():
    lo, hi = JOIN_MASS
    rows = curve_rows(3, range(0, 12), lambda a: lo if a % 2 else hi) + curve_rows(2, range(0, 12), 1.0)
    run, = runs_of(rows)
    assert run.u_hi - run.u_lo == 0.5 * 12

def test_the_mass_range_is_a_parameter():
    rows = curve_rows(3, range(0, 12), 3.0) + curve_rows(2, range(0, 12), 1.0)
    assert runs_of(rows) == []
    assert len(runs_of(rows, mass_range=(2.5, 3.5))) == 1

def test_only_one_sheet_rows_are_looked_at():
    rows = scene() + [row(3, a, 2.0, m=2) for a in range(30, 45)] + [row(5, a, 2.0, m=3) for a in range(0, 12)]
    assert [r.k for r in runs_of(rows)] == [3]

def test_the_status_of_a_row_selects_it_by_its_mass():
    heavy = curve_rows(3, range(0, 12), 2.0, status="short") + curve_rows(2, range(0, 12), 1.0)
    assert runs_of(heavy) == []                                 # short: no valid mass
    inclined = curve_rows(3, range(0, 12), 2.0, status="inclined") + curve_rows(2, range(0, 12), 1.0)
    assert runs_of(inclined) == []

def test_a_degenerate_band_does_not_matter_to_a_join():
    rows = (curve_rows(3, range(0, 12), 2.0, status="band_nan") + curve_rows(2, range(0, 12), 1.0, status="band_nan")
            + curve_rows(4, range(0, 12), 1.0))
    run, = runs_of(rows)
    assert run.k == 3 and run.u_hi - run.u_lo == 0.5 * 12

def test_the_clean_flag_is_not_a_condition_of_a_join():
    rows = curve_rows(3, range(0, 12), 2.0, clean=False) + curve_rows(2, range(0, 12), 1.0, clean=False)
    assert len(runs_of(rows)) == 1

def test_each_winding_has_its_own_runs():
    rows = scene() + curve_rows(6, range(0, 12), 2.0) + curve_rows(5, range(0, 12), 1.0)
    assert [r.k for r in runs_of(rows)] == [3, 6]

def test_a_run_ends_where_the_winding_stops_being_heavy():
    rows = curve_rows(3, range(0, 30), lambda a: 2.0 if 8 <= a < 22 else 1.0) + curve_rows(2, range(0, 30), 1.0)
    run, = runs_of(rows)
    assert (run.u_lo, run.u_hi) == (0.5 * 8 - ARC_MM / 2, 0.5 * 21 + ARC_MM / 2)
    assert (run.theta_lo, run.theta_hi) == (0.25 * 8, 0.25 * 21)


# --- the neighbours -----------------------------------------------------------------------------

def test_both_neighbours_normal_keep_the_run():
    assert len(runs_of(scene(below=1.0, above=1.0))) == 1

def test_a_heavy_neighbour_vetoes_the_run_whichever_side():
    assert runs_of(scene(below=2.0, above=1.0)) == []
    assert runs_of(scene(below=1.0, above=2.0)) == []

def test_a_light_neighbour_vetoes_the_run_as_well():
    assert runs_of(scene(below=0.5, above=1.0)) == []

def test_one_neighbour_with_no_data_and_the_other_normal_keeps_the_run():
    assert len(runs_of(scene(above=1.0, below=None))) == 1
    assert len(runs_of(scene(above=None, below=1.0))) == 1
    # arcs of the curve 2 that are heavy, but at angles (4 rad) outside the run (0 .. 2.75 rad): no data over the run
    rows = scene(below=None) + curve_rows(2, range(40, 52), 3.0, theta=4.0)
    assert [r.k for r in runs_of(rows)] == [3]

def test_a_neighbour_with_no_data_cannot_validate_the_run():
    assert runs_of(scene(below=None, above=None)) == []
    rows = (scene(below=None, above=None) + curve_rows(2, range(40, 52), 1.0, theta=4.0)
            + curve_rows(4, range(40, 52), 1.0, theta=4.0))
    assert runs_of(rows) == []                                  # normal arcs, but outside the angles of the run

def test_one_neighbour_with_no_data_does_not_excuse_a_heavy_one():
    assert runs_of(scene(below=None, above=2.0)) == []
    assert runs_of(scene(below=2.0, above=None)) == []

def test_a_neighbour_that_does_not_exist_is_simply_absent():
    first = curve_rows(0, range(0, 12), 2.0) + curve_rows(1, range(0, 12), 1.0)                 # no curve -1
    assert [r.k for r in runs_of(first)] == [0]
    last = curve_rows(5, range(0, 12), 2.0) + curve_rows(4, range(0, 12), 1.0)                  # no curve 6
    assert [r.k for r in runs_of(last)] == [5]
    assert runs_of(curve_rows(0, range(0, 12), 2.0)) == []  # nobody says it is normal
    assert runs_of(curve_rows(0, range(0, 12), 2.0) + curve_rows(1, range(0, 12), 2.0)) == []   # ... or abnormal

def test_neighbour_bounds_are_inclusive():
    lo, hi = JOIN_NEIGHBOUR_NORMAL
    for ratio in (lo, hi):
        assert len(runs_of(scene(below=ratio, above=1.0))) == 1
        assert len(runs_of(scene(below=1.0, above=ratio))) == 1
    for ratio in (np.nextafter(lo, 0.0), np.nextafter(hi, 9.0)):
        assert runs_of(scene(below=ratio, above=1.0)) == []
        assert runs_of(scene(below=1.0, above=ratio)) == []

def test_the_neighbour_range_is_a_parameter():
    assert runs_of(scene(below=2.0, above=2.0)) == []
    assert [r.k for r in runs_of(scene(below=2.0, above=2.0), neighbour_range=(1.9, 2.1))] == [2, 3, 4]

def test_the_neighbour_is_judged_on_its_median():
    # 7 normal arcs and 5 very heavy ones: the median is normal (the mean, 1.83, would not be)
    rows = scene(above=None) + curve_rows(4, range(0, 12), lambda a: 1.0 if a < 7 else 3.0)
    assert len(runs_of(rows)) == 1
    rows = scene(above=None) + curve_rows(4, range(0, 12), lambda a: 3.0 if a < 7 else 1.0)   # now the median is heavy
    assert runs_of(rows) == []

def test_only_the_neighbour_arcs_inside_the_angles_of_the_run_count():
    base = scene(above=None)                                    # the run (0 .. 2.75 rad), curve 2 normal
    heavy = lambda theta: [row(4, 50 + i, 3.0, theta=theta) for i in range(30)]
    assert len(runs_of(base + curve_rows(4, range(20, 40), 3.0, theta=4.0))) == 1     # at 4 rad: ignored
    assert len(runs_of(base + heavy(0.25 * 11 + 0.01))) == 1                # just after the last arc: ignored
    assert len(runs_of(base + heavy(-0.01))) == 1                           # just before the first arc: ignored
    assert runs_of(base + heavy(0.25 * 11 - 0.01)) == []                    # just inside: they veto
    assert runs_of(base + heavy(0.01)) == []

@pytest.mark.parametrize("edge", [0.0, 0.25 * 11], ids=["first arc", "last arc"])
def test_the_angles_of_the_run_are_inclusive_at_both_ends(edge):
    on_edge = [row(4, 50 + i, 3.0, theta=edge) for i in range(20)]
    assert runs_of(scene(above=None) + on_edge) == []           # 20 heavy arcs at exactly one end of the angles count


def test_neighbours_that_are_not_mass_valid_or_not_one_sheet_do_not_count():
    junk = curve_rows(4, range(0, 12), 3.0, status="short") + curve_rows(4, range(0, 12), 3.0, m=2)
    assert len(runs_of(scene(below=1.0, above=None) + junk)) == 1           # they cannot veto
    assert runs_of(scene(below=None, above=None) + junk) == []              # ... nor validate

def test_a_neighbour_with_a_degenerate_band_is_judged_on_its_mass():
    rows = scene(above=None) + curve_rows(4, range(0, 12), 3.0, status="band_nan")
    assert runs_of(rows) == []
    rows = scene(below=None, above=None) + curve_rows(4, range(0, 12), 1.0, status="band_nan")
    assert len(runs_of(rows)) == 1


# --- the angle of the run goes through 2 pi -----------------------------------------------------

def across_two_pi(neighbour_theta, neighbour_ratio=1.0, other=None):
    """A join of 12 arcs from 6.0 to 8.75 rad (2.75 rad = 0.25 rad per arc) mod 2 pi, and the arcs of curve 4."""
    rows = [row(3, a, 2.0, theta=6.0 + 0.25 * a) for a in range(0, 12)]
    rows += [row(4, 100 + i, neighbour_ratio, theta=t) for i, t in enumerate(neighbour_theta)]
    if other is not None:
        rows += [row(2, 200 + i, other[1], theta=t) for i, t in enumerate(other[0])]
    return rows

def test_a_run_through_two_pi_is_found_with_its_neighbours_on_both_sides_of_the_turn():
    rows = across_two_pi([6.1, 6.2, 0.1, 0.5, 1.0, 2.0])        # normal arcs, before and after the turn
    run, = runs_of(rows)
    assert run.theta_lo == pytest.approx(6.0) and run.theta_hi == pytest.approx((6.0 + 0.25 * 11) % TWO_PI)
    assert run.theta_lo > run.theta_hi                          # the interval goes through 2 pi

def test_neighbour_arcs_after_the_turn_are_inside_the_interval():
    rows = across_two_pi([0.1, 0.5, 1.0, 2.0])                  # only after the turn: still data
    assert len(runs_of(rows)) == 1
    heavy = across_two_pi([0.1, 0.5, 1.0, 2.0], neighbour_ratio=3.0)
    assert runs_of(heavy) == []

def test_neighbour_arcs_before_the_turn_are_inside_the_interval():
    assert len(runs_of(across_two_pi([6.05, 6.2]))) == 1
    assert runs_of(across_two_pi([6.05, 6.2], neighbour_ratio=3.0)) == []

def test_neighbour_arcs_outside_the_interval_through_two_pi_are_ignored():
    # 3 .. 5.9 rad are in the complement of [6.0, 8.75 mod 2 pi = 2.47]: the heavy arcs there do not count
    rows = across_two_pi([0.1, 6.1], other=([3.0, 3.5, 4.0, 5.0, 5.9] * 3, 3.0))
    assert len(runs_of(rows)) == 1
    rows = across_two_pi([3.0, 4.0, 5.0], other=([3.0, 3.5, 4.0], 1.0))
    assert runs_of(rows) == []                                  # nothing inside: nothing validates

LAST_THETA = (6.0 + 0.25 * 11) % TWO_PI                        # the last arc of across_two_pi

@pytest.mark.parametrize("edge, outside", [(6.0, 6.0 - 0.01), (LAST_THETA, LAST_THETA + 0.01)],
                         ids=["first arc", "last arc"])
def test_the_ends_of_an_interval_through_two_pi_are_inclusive(edge, outside):
    validated = ([6.1], 1.0)                                    # one normal arc of curve 2 inside the interval
    at_end = across_two_pi([edge] * 10, neighbour_ratio=3.0, other=validated)
    beyond = across_two_pi([outside] * 10, neighbour_ratio=3.0, other=validated)
    assert runs_of(at_end) == []                                # heavy arcs at the end of the interval: veto
    assert len(runs_of(beyond)) == 1                            # just beyond: ignored


# --- companion slices -----------------------------------------------------------------------------

def run_at(k=3, u_lo=10.0, u_hi=16.0):
    return JoinRun(k, u_lo, u_hi, 1.0, 2.0)

def test_no_companion_slice_confirms_nothing():
    assert confirm_joins([run_at()], []) == []

def test_one_companion_slice_that_has_the_run_confirms_it():
    run = run_at()
    assert confirm_joins([run], [[run]]) == [run]
    assert confirm_joins([run], [[run_at(u_lo=11.0, u_hi=17.0)]]) == [run]

def test_every_companion_slice_must_have_the_run():
    run = run_at()
    assert confirm_joins([run], [[run], [run], [run]]) == [run]
    assert confirm_joins([run], [[run], [run], []]) == []
    assert confirm_joins([run], [[], [run], [run]]) == []
    assert confirm_joins([run], [[run], [run_at(k=4)]]) == []

def test_a_companion_run_on_another_winding_does_not_confirm():
    assert confirm_joins([run_at(k=3)], [[run_at(k=2)], [run_at(k=3)]]) == []

def test_the_companion_run_may_be_just_inside_the_matching_distance():
    run = run_at(u_lo=10.0, u_hi=16.0)
    margin = JOIN_MATCH_MM
    assert margin == 3.0
    assert confirm_joins([run], [[run_at(u_lo=16.0 + margin, u_hi=30.0)]]) == [run]          # touches the end
    assert confirm_joins([run], [[run_at(u_lo=16.0 + margin + 0.01, u_hi=30.0)]]) == []      # just beyond
    assert confirm_joins([run], [[run_at(u_lo=0.0, u_hi=10.0 - margin)]]) == [run]           # touches the start
    assert confirm_joins([run], [[run_at(u_lo=0.0, u_hi=10.0 - margin - 0.01)]]) == []

def test_the_matching_distance_is_a_parameter():
    run = run_at(u_lo=10.0, u_hi=16.0)
    far = run_at(u_lo=18.0, u_hi=30.0)
    assert confirm_joins([run], [[far]], match_mm=2.0) == [run]
    assert confirm_joins([run], [[far]], match_mm=1.99) == []
    assert confirm_joins([run], [[far]]) == [run]               # JOIN_MATCH_MM = 3.0
    assert confirm_joins([run], [[run_at(u_lo=16.0, u_hi=20.0)]], match_mm=0.0) == [run]    # touching
    assert confirm_joins([run], [[run_at(u_lo=16.01, u_hi=20.0)]], match_mm=0.0) == []

def test_a_companion_run_that_contains_or_is_inside_the_run_matches():
    run = run_at(u_lo=10.0, u_hi=16.0)
    assert confirm_joins([run], [[run_at(u_lo=0.0, u_hi=100.0)]]) == [run]
    assert confirm_joins([run], [[run_at(u_lo=12.0, u_hi=13.0)]]) == [run]

def test_any_of_the_runs_of_a_companion_slice_may_be_the_match():
    run = run_at()
    companion = [run_at(k=3, u_lo=50.0, u_hi=60.0), run_at(k=2), run_at(k=3, u_lo=9.0, u_hi=12.0)]
    assert confirm_joins([run], [companion]) == [run]

def test_confirmation_keeps_the_order_and_drops_the_runs_that_do_not_match():
    a, b, c = run_at(k=5), run_at(k=3, u_lo=40.0, u_hi=50.0), run_at(k=3)
    companions = [[c, a], [run_at(k=5, u_lo=12.0, u_hi=20.0), c]]
    assert confirm_joins([a, b, c], companions) == [a, c]
    assert confirm_joins([c, b, a], companions) == [c, a]
    assert confirm_joins([], companions) == []

def test_confirmation_does_not_modify_its_arguments():
    run = run_at()
    runs, companions = [run], [[run]]
    confirm_joins(runs, companions)
    assert runs == [run] and companions == [[run]]


# --- in_join and expected_layers --------------------------------------------------------------

def test_in_join_is_inclusive_at_both_ends_and_on_the_winding_of_the_join():
    joins = [run_at(k=3, u_lo=10.0, u_hi=16.0)]
    assert in_join(3, 10.0, joins) is True
    assert in_join(3, 16.0, joins) is True
    assert in_join(3, 13.0, joins) is True
    assert in_join(3, np.nextafter(10.0, 0.0), joins) is False
    assert in_join(3, np.nextafter(16.0, 99.0), joins) is False
    assert in_join(2, 13.0, joins) is False
    assert in_join(4, 13.0, joins) is False

def test_in_join_with_no_join_or_no_position():
    assert in_join(3, 13.0, []) is False
    assert in_join(3, float("nan"), [run_at()]) is False

def test_in_join_looks_at_every_join_and_takes_numpy_integers():
    joins = [run_at(k=3, u_lo=10.0, u_hi=16.0), run_at(k=3, u_lo=30.0, u_hi=32.0), run_at(k=5)]
    assert in_join(np.int64(3), 31.0, joins) is True
    assert in_join(np.int64(5), np.float64(12.0), joins) is True
    assert in_join(3, 20.0, joins) is False

def test_expected_layers_is_m_plus_the_joins_the_path_crosses():
    joins = [run_at(k=3, u_lo=10.0, u_hi=16.0), run_at(k=4, u_lo=20.0, u_hi=26.0)]
    assert expected_layers(2, 1, [5.0], joins) == 1             # sheet 2 is not joined
    assert expected_layers(3, 1, [12.0], joins) == 2            # one sheet, in the join
    assert expected_layers(3, 1, [18.0], joins) == 1            # one sheet, outside it
    assert expected_layers(2, 3, [5.0, 12.0, 30.0], joins) == 4          # enters sheet 3 inside the join
    assert expected_layers(2, 3, [5.0, 12.0, 22.0], joins) == 5          # ... and sheet 4 inside its join
    assert expected_layers(2, 3, [5.0, 18.0, 22.0], joins) == 4          # sheet 3 outside, sheet 4 inside
    assert expected_layers(3, 2, [12.0, 22.0], joins) == 4
    assert expected_layers(3, 2, [12.0, 22.0], []) == 2

def test_expected_layers_gives_a_plain_int():
    joins = [run_at()]
    assert type(expected_layers(3, 2, [12.0, 5.0], joins)) is int
    assert type(expected_layers(np.int64(3), np.int64(1), np.array([12.0]), joins)) is int

def test_expected_layers_needs_a_position_per_sheet():
    with pytest.raises(ValueError, match="u_path"):
        expected_layers(3, 3, [12.0, 13.0], [run_at()])
    assert expected_layers(3, 1, [12.0, 99.0, 99.0], [run_at()]) == 2        # extra positions are not looked at

def test_expected_layers_rejects_a_non_positive_number_of_sheets():
    with pytest.raises(ValueError, match="m"):
        expected_layers(3, 0, [], [run_at()])


# --- the final reference -----------------------------------------------------------------------

def test_the_final_reference_is_the_median_of_the_clean_arcs_outside_the_joins():
    join = run_at(k=3, u_lo=10.0, u_hi=16.0)
    rows = (curve_rows(1, range(0, 3), 1.0)                                                         # 8.0, kept
            + [row(3, 100 + i, 2.0, u=13.0) for i in range(10)]                                     # clean, in the join
            + [row(3, 200 + i, 7.0, u=40.0, clean=False) for i in range(10)])                       # not clean
    assert sigma_ref_final(rows, [join]) == 1.0 * SIGMA
    assert sigma_ref_final(rows, []) == 2.0 * SIGMA             # without the join, its arcs weigh in

def test_the_final_reference_keeps_the_arcs_with_a_degenerate_band():
    rows = (curve_rows(1, range(0, 2), 1.0) + curve_rows(1, range(2, 5), 1.5, status="band_nan"))
    assert sigma_ref_final(rows, []) == 1.5 * SIGMA             # 2 "ok" and 3 "band_nan": the band_nan arcs count

def test_the_final_reference_leaves_out_what_has_no_valid_mass_or_is_not_one_sheet():
    rows = (curve_rows(1, range(0, 3), 1.0)
            + curve_rows(1, range(3, 13), 9.0, status="short") + curve_rows(1, range(13, 23), 9.0, status="inclined")
            + curve_rows(1, range(23, 33), 9.0, m=2))
    assert sigma_ref_final(rows, []) == 1.0 * SIGMA

def test_the_final_reference_looks_at_the_winding_of_each_join():
    rows = curve_rows(1, range(0, 3), 1.0) + [row(2, 10 + i, 3.0, u=13.0) for i in range(10)]
    assert sigma_ref_final(rows, [run_at(k=2)]) == 1.0 * SIGMA
    assert sigma_ref_final(rows, [run_at(k=3)]) == 3.0 * SIGMA   # the join is on another winding: nothing excluded

def test_the_final_reference_needs_a_clean_arc_outside_the_joins():
    with pytest.raises(ValueError, match="clean"):
        sigma_ref_final([], [])
    with pytest.raises(ValueError, match="clean"):
        sigma_ref_final(curve_rows(1, range(0, 5), 1.0, clean=False), [])
    with pytest.raises(ValueError, match="clean"):
        sigma_ref_final([row(3, i, 2.0, u=13.0) for i in range(5)], [run_at(k=3)])
    with pytest.raises(ValueError, match="clean"):
        sigma_ref_final(curve_rows(1, range(0, 5), 1.0, status="short") + curve_rows(1, range(0, 5), 1.0, m=2), [])


# --- the arguments ------------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_ref", [0.0, -1.0, float("nan"), float("inf")])
def test_the_reference_mass_must_be_a_positive_finite_number(sigma_ref):
    with pytest.raises(ValueError, match="sigma_ref"):
        find_join_runs(scene(), sigma_ref)

@pytest.mark.parametrize("kw", [{"arc_mm": 0.0}, {"arc_mm": float("nan")}, {"min_mm": -1.0}, {"min_mm": float("nan")},
                                {"mass_range": (2.4, 1.6)}, {"neighbour_range": (1.4, 0.6)}])
def test_the_parameters_of_the_rule_are_validated(kw):
    with pytest.raises(ValueError):
        runs_of(scene(), **kw)

@pytest.mark.parametrize("match_mm", [-0.1, float("nan"), float("inf")])
def test_the_matching_distance_is_validated(match_mm):
    with pytest.raises(ValueError, match="match_mm"):
        confirm_joins([run_at()], [[run_at()]], match_mm=match_mm)

def test_no_rows_no_runs():
    assert runs_of([]) == []
    assert runs_of(curve_rows(3, range(0, 12), 1.0)) == []      # nothing heavy
