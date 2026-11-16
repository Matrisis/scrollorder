import numpy as np
import pytest

from scrollorder.experiments import evaluate_barcode, evaluate_ordered_cut, wrap_accuracy
from scrollorder.phantom import make_phantom
from scrollorder.polar import polar_resample

DR = 0.5


def _polar(ph):
    stack = ph["stack"]
    return polar_resample(ph["image"], ph["center_xy"], stack.shape[1], stack.max() + 5, DR)


def test_wrap_accuracy_is_one_on_the_truth_and_zero_on_the_neighbouring_wrap():
    truth = np.array([[10.0, 10.0], [20.0, 21.0], [30.0, 33.0], [40.0, 44.0]])  # anchors + 2 sheets
    assert wrap_accuracy(truth[1:3], truth).mean() == 1.0
    assert wrap_accuracy(truth[1:3] + 2.0, truth).mean() == 1.0   # small error, still the right wrap
    assert wrap_accuracy(truth[2:4], truth).mean() == 0.0          # every sheet one wrap too far


@pytest.fixture(scope="module")
def clean_rows():
    ph = make_phantom(seed=0, noise=0.0, hole_fraction=0.0, compress=0.0)
    return evaluate_ordered_cut(_polar(ph), DR, ph["stack"], k_values=[4], theta_stride=4, anchor_stride=8)


def test_ordered_cut_reports_one_row_per_method(clean_rows):
    assert {r["method"] for r in clean_rows} == {"interp", "snap", "cut"}
    assert all(r["K"] == 4 and r["n_samples"] > 0 for r in clean_rows)


def test_ordered_cut_finds_the_sheets_on_a_clean_phantom(clean_rows):
    cut = next(r for r in clean_rows if r["method"] == "cut")
    assert cut["accuracy"] > 0.95


def test_ordered_cut_never_merges_two_sheets():
    ph = make_phantom(seed=1)  # compressed sector, holes and noise
    rows = evaluate_ordered_cut(_polar(ph), DR, ph["stack"], k_values=[6], theta_stride=4, anchor_stride=10)
    cut = next(r for r in rows if r["method"] == "cut")
    assert cut["merged"] == 0.0


def test_ordered_cut_skips_columns_where_the_truth_is_missing():
    ph = make_phantom(seed=2, noise=0.0, hole_fraction=0.0, compress=0.0)
    stack = ph["stack"].copy()
    stack[:, 100:400] = np.nan
    full = evaluate_ordered_cut(_polar(ph), DR, ph["stack"], [4], theta_stride=4, anchor_stride=12)
    part = evaluate_ordered_cut(_polar(ph), DR, stack, [4], theta_stride=4, anchor_stride=12)
    assert 0 < part[0]["n_samples"] < full[0]["n_samples"]


def test_barcode_evaluation_reports_both_methods_for_each_band_width():
    ph = make_phantom(seed=3)
    rows = evaluate_barcode(ph["stack"], band_cols=[4, 120], n_trials=40, seed=0)
    assert [(r["band_cols"], r["method"]) for r in rows] == [
        (4, "nearest"), (4, "barcode"), (120, "nearest"), (120, "barcode")]
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
    assert rows[0]["band_deg"] == pytest.approx(2.0)


def test_barcode_beats_nearest_radius_across_a_wide_band_on_the_phantom():
    ph = make_phantom(seed=4, pitch_sigma=0.3)
    rows = evaluate_barcode(ph["stack"], band_cols=[120], n_trials=80, seed=1)
    nearest, barcode = rows
    assert barcode["accuracy"] > nearest["accuracy"] + 0.3
    assert barcode["accuracy"] > 0.9


def test_cut_falls_back_to_equal_division_where_the_image_shows_nothing():
    ph = make_phantom(seed=5)
    blank = np.zeros_like(_polar(ph))
    rows = evaluate_ordered_cut(blank, DR, ph["stack"], [5], theta_stride=8, anchor_stride=10)
    by = {r["method"]: r for r in rows}
    assert by["cut"]["accuracy"] == by["interp"]["accuracy"]


def test_a_lone_sheet_with_holes_is_not_pulled_onto_the_anchor_ridges():
    ph = make_phantom(seed=0)  # holes in the ridges, noise, bright anchors on both sides
    rows = evaluate_ordered_cut(_polar(ph), DR, ph["stack"], [2], theta_stride=2)
    by = {r["method"]: r for r in rows}
    assert by["cut"]["accuracy"] > 0.97


def test_sheets_with_holes_are_not_pulled_onto_their_neighbours_ridges():
    ph = make_phantom(seed=0)  # equal division happens to be right here; the image must not make it worse
    rows = evaluate_ordered_cut(_polar(ph), DR, ph["stack"], [5], theta_stride=2)
    by = {r["method"]: r for r in rows}
    assert by["cut"]["accuracy"] > 0.95
    assert by["cut"]["accuracy"] >= by["snap"]["accuracy"]


def test_the_prior_does_not_drag_a_whole_bundle_onto_the_wrong_wraps():
    ph = make_phantom(seed=0)  # here equal division is off by more than half a spacing for many sheets
    rows = evaluate_ordered_cut(_polar(ph), DR, ph["stack"], [20], theta_stride=4, anchor_stride=10)
    by = {r["method"]: r for r in rows}
    assert by["interp"]["accuracy"] < 0.8
    assert by["cut"]["accuracy"] > 0.9
