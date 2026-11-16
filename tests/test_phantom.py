import numpy as np

from scrollorder.phantom import make_phantom
from scrollorder.polar import polar_resample


def test_sheets_are_strictly_ordered_and_never_touch():
    ph = make_phantom(seed=0)
    gaps = np.diff(ph["stack"], axis=0)
    assert gaps.min() > 1.5


def test_stack_is_one_continuous_spiral():
    ph = make_phantom(seed=1)
    stack = ph["stack"]
    step = np.abs(np.diff(stack, axis=1)).max()
    seam = np.abs(stack[1:, 0] - stack[:-1, -1]).max()
    assert seam <= 2 * step + 1e-9


def test_spacing_between_sheets_is_irregular():
    ph = make_phantom(seed=2)
    gaps = np.diff(ph["stack"][:, 0])
    assert gaps.std() / gaps.mean() > 0.1


def test_image_is_bright_on_the_sheets_and_dark_between_them():
    ph = make_phantom(seed=3, hole_fraction=0.0, noise=0.0)
    stack = ph["stack"]
    n_theta = stack.shape[1]
    polar = polar_resample(ph["image"], ph["center_xy"], n_theta, r_max=stack.max() + 5, dr=0.5)
    cols = np.arange(n_theta)
    wide = np.diff(stack, axis=0) > 6.0  # only where sheets are clearly separated
    on = polar[cols, np.rint(stack[:-1] / 0.5).astype(int)][wide]
    mid = polar[cols, np.rint((stack[:-1] + stack[1:]) / 2 / 0.5).astype(int)][wide]
    assert on.mean() > 0.8 and mid.mean() < 0.2


def test_same_seed_gives_the_same_phantom():
    a, b = make_phantom(seed=4), make_phantom(seed=4)
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["stack"], b["stack"])
