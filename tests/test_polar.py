import numpy as np

from scrollorder.polar import polar_resample, positions_to_radius, sample_between, stack_from_unwrapped


def test_polar_resample_puts_a_circle_on_a_constant_radius():
    yy, xx = np.mgrid[0:128, 0:128]
    center = (64.5, 50.2)
    rho = np.hypot(xx - center[0], yy - center[1])
    image = np.exp(-((rho - 40.0) ** 2) / 4.0)
    polar = polar_resample(image, center, n_theta=90, r_max=48.0, dr=0.5)
    assert polar.shape == (90, 96)
    assert np.allclose(polar.argmax(axis=1) * 0.5, 40.0, atol=0.5)


def test_polar_resample_angle_zero_points_along_plus_x():
    image = np.zeros((64, 64))
    image[32, 50] = 1.0  # 18 px to the right of the centre (x=32, y=32)
    polar = polar_resample(image, (32.0, 32.0), n_theta=8, r_max=30.0)
    assert polar[0].argmax() == 18
    assert polar[0].max() > 0.9 and polar[4].max() < 0.1


def test_stack_from_unwrapped_cuts_one_row_per_winding():
    stack = stack_from_unwrapped(np.arange(24.0), n_theta=8)
    assert stack.shape == (3, 8)
    assert stack[1, 0] == 8.0 and stack[2, 7] == 23.0


def test_sample_between_reads_evenly_spaced_radii_strictly_inside_the_anchors():
    polar = np.tile(np.arange(100.0) * 0.5, (4, 1))  # value == radius, dr = 0.5
    r_lo, r_hi = np.full(4, 10.0), np.array([20.0, 20.0, 30.0, 30.0])
    sampled = sample_between(polar, 0.5, r_lo, r_hi, n_pos=9)
    assert sampled.shape == (4, 9)
    assert np.allclose(sampled[0], np.arange(11.0, 20.0))
    assert np.allclose(sampled[2], 10.0 + 2.0 * np.arange(1, 10))


def test_positions_to_radius_inverts_the_sampling_grid():
    r_lo, r_hi = np.array([10.0, 10.0]), np.array([20.0, 30.0])
    radius = positions_to_radius(np.array([[0, 0], [8, 8]]), r_lo, r_hi, n_pos=9)
    assert np.allclose(radius, [[11.0, 12.0], [19.0, 28.0]])


def test_sample_between_can_include_the_anchor_positions():
    polar = np.tile(np.arange(100.0) * 0.5, (2, 1))
    sampled = sample_between(polar, 0.5, np.full(2, 10.0), np.full(2, 20.0), n_pos=9, with_anchors=True)
    assert sampled.shape == (2, 11)
    assert np.allclose(sampled[0], np.arange(10.0, 21.0))


def test_ridge_evidence_keeps_only_the_crest_of_each_ridge():
    from scrollorder.polar import ridge_evidence

    profile = np.array([[0.0, 0.1, 0.5, 1.0, 0.5, 0.1, 0.0, 0.3, 0.6, 0.3, 0.0]])
    assert ridge_evidence(profile).tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0]]


def test_ridge_evidence_marks_the_middle_of_a_flat_blob():
    from scrollorder.polar import ridge_evidence

    profile = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    assert ridge_evidence(profile).tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]


def test_ridge_evidence_ignores_the_ridges_of_the_anchors_themselves():
    from scrollorder.polar import ridge_evidence

    # the first and last samples sit on the anchors: their flank or blob is not a new sheet
    profile = np.array([[1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.6, 0.9]])
    assert ridge_evidence(profile).tolist() == [[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]]
