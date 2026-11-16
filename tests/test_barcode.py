import numpy as np
import pytest

from scrollorder.barcode import estimate_shift


def _irregular_stack(seed=0, n=40, pitch=10.0, sigma=0.3):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(pitch * np.exp(sigma * rng.standard_normal(n)))


def test_recovers_shift_when_inner_sheets_are_missing_on_the_right():
    left = _irregular_stack()
    est = estimate_shift(left, left[2:])
    assert est.shift == 2  # right[j] is the same sheet as left[j + 2]


def test_recovers_negative_shift_when_right_has_extra_inner_sheets():
    left = _irregular_stack(1)
    right = np.concatenate([[left[0] - 21.0, left[0] - 9.0], left])
    assert estimate_shift(left, right).shift == -2


def test_barcode_survives_a_radial_drift_that_fools_nearest_radius():
    left = _irregular_stack(2)
    right = left[1:] + 26.0  # the whole stack moved by ~2.6 sheet spacings
    assert estimate_shift(left, right, method="nearest").shift != 1
    assert estimate_shift(left, right, method="barcode").shift == 1


def test_barcode_survives_uniform_compression_of_the_stack():
    left = _irregular_stack(3)
    right = 0.8 * left[1:] + 15.0
    assert estimate_shift(left, right).shift == 1


def test_tolerates_measurement_noise_on_both_stacks():
    rng = np.random.default_rng(4)
    left = _irregular_stack(4)
    right = left[3:] + 12.0 + rng.normal(0, 0.5, left.size - 3)
    assert estimate_shift(left + rng.normal(0, 0.5, left.size), right).shift == 3


def test_irregular_spacing_gives_high_confidence():
    left = _irregular_stack(5)
    assert estimate_shift(left, left[1:] + 7.0).confidence > 0.5


def test_regular_spacing_is_reported_as_ambiguous_and_falls_back_to_continuity():
    left = 100.0 + 10.0 * np.arange(30)
    est = estimate_shift(left, left[1:] + 4.0)
    assert est.confidence < 0.2  # a perfectly regular barcode carries no information
    assert est.shift == 1        # the weak continuity prior still picks the closest match


def test_refuses_to_decide_from_too_few_sheets():
    with pytest.raises(ValueError, match="at least"):
        estimate_shift(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
