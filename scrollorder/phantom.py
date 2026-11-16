"""Synthetic scroll cross-section with known answer.  FOR UNIT TESTS ONLY.

It checks that the code is correct; it says nothing about real scrolls.  Any
claim of usefulness must be measured on real scroll data (see README).
"""
from __future__ import annotations

import numpy as np

from .polar import stack_from_unwrapped


def make_phantom(
    seed: int = 0,
    n_windings: int = 30,
    n_theta: int = 720,
    r0: float = 40.0,
    pitch: float = 9.0,
    pitch_sigma: float = 0.25,
    squash: float = 0.15,
    compress: float = 0.5,
    compress_width: float = 0.5,
    ridge_sigma: float = 1.2,
    hole_fraction: float = 0.1,
    noise: float = 0.15,
) -> dict:
    """Return {"image", "center_xy", "stack"}: a squashed, locally compressed spiral.

    stack[k, i] is the true radius of winding k along ray i.  Sheet spacing is
    irregular (log-normal pitch), shrinks by `compress` in one angular sector,
    ridges have holes, and the image is noisy -- so neighbouring sheets merge
    visually in the compressed sector, as they do in real scans.
    """
    rng = np.random.default_rng(seed)
    n = n_windings * n_theta
    turns = np.arange(n) / n_theta
    theta = 2 * np.pi * (turns % 1.0)

    knots = pitch * np.clip(np.exp(pitch_sigma * rng.standard_normal(n_windings + 2)), 0.55, 2.0)
    local_pitch = np.interp(turns, np.arange(n_windings + 2), knots)
    radius = r0 + np.cumsum(local_pitch) / n_theta

    theta_c, theta_s = rng.uniform(0, 2 * np.pi, size=2)
    bump = np.exp(-(((theta - theta_c + np.pi) % (2 * np.pi) - np.pi) ** 2) / (2 * compress_width**2))
    radius = r0 + (radius - r0) * (1.0 - compress * bump)       # compressed sector
    radius = radius * (1.0 + squash * np.cos(2 * (theta - theta_s)))  # squashed scroll
    stack = stack_from_unwrapped(radius, n_theta)

    amplitude = np.ones_like(stack)
    for k in range(n_windings):                                   # holes in the ridges
        for _ in range(int(round(hole_fraction * n_theta / 40.0))):
            start = rng.integers(0, n_theta)
            amplitude[k, np.arange(start, start + rng.integers(20, 60)) % n_theta] = 0.0

    half = int(np.ceil(stack.max())) + 8
    center_xy = (half + 0.3, half - 0.4)
    yy, xx = np.mgrid[0 : 2 * half + 1, 0 : 2 * half + 1]
    rho = np.hypot(xx - center_xy[0], yy - center_xy[1]).ravel()
    t = (np.arctan2(yy - center_xy[1], xx - center_xy[0]).ravel() % (2 * np.pi)) / (2 * np.pi) * n_theta
    i0 = np.floor(t).astype(int) % n_theta
    i1 = (i0 + 1) % n_theta
    w = t - np.floor(t)
    image = np.zeros(rho.size)
    for k in range(n_windings):
        # across the seam (i1 == 0) the curve continues on the next winding
        nxt = np.where(i1 == 0, stack[min(k + 1, n_windings - 1), 0], stack[k, i1])
        d = rho - ((1 - w) * stack[k, i0] + w * nxt)
        image = np.maximum(image, amplitude[k, i0] * np.exp(-(d**2) / (2 * ridge_sigma**2)))
    image = image.reshape(yy.shape)
    if noise > 0:
        image = image + noise * rng.standard_normal(image.shape)
    return {"image": np.clip(image, 0.0, 1.0), "center_xy": center_xy, "stack": stack}


# --- a 3D roll of closed rings with a known sheet mass ---------------------------------------

_TWO_PI = 2 * np.pi
_TABLE_NODES = 8192          # at least this many angles in the ring tables (a multiple of n_cols)


def _wrapped(theta, center):
    """Signed angular distance from `center`, in [-pi, pi)."""
    return (np.asarray(theta, dtype=float) - center + np.pi) % _TWO_PI - np.pi


def _pressed_bump(theta, center, width):
    """Raised cosine centred on `center`, 1 there, 0 at +-width/2 and beyond: where two sheets touch."""
    d = np.abs(_wrapped(theta, center))
    half = width / 2.0
    return np.where(d < half, 0.5 * (1.0 + np.cos(np.pi * d / half)), 0.0)


def _join_plateau(theta, center, width, flank=0.2):
    """1 over `width` rad around `center`, then a cosine flank of `flank` rad on each side, 0 beyond."""
    d = np.abs(_wrapped(theta, center))
    half = width / 2.0
    ramp = 0.5 * (1.0 + np.cos(np.pi * (d - half) / flank))
    return np.where(d <= half, 1.0, np.where(d >= half + flank, 0.0, ramp))


def make_volume_phantom(
    n_windings: int = 8,
    r0: float = 30.0,
    pitch: float = 12.0,
    thickness: float = 4.0,
    pressed_gaps: tuple[int, int] | None = (2, 5),
    pressed_theta: float = np.pi / 2,
    pressed_width: float = 1.6,
    join_winding: int | None = 3,
    join_theta: float = 3 * np.pi / 2,
    join_width: float = 1.8,
    sheet_grey: float = 200.0,
    air_grey: float = 20.0,
    n_z: int = 13,
    n_cols: int = 720,
    px_mm: float = 0.05,
    noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """A 3D roll of closed rings, the same in every plane, whose sheet mass is known.  FOR UNIT TESTS ONLY.

    Winding k is the sheet [recto_k(theta), recto_k(theta) + radial_thickness_k(theta)], with
    recto_0 = r0 and recto_{k+1} = recto_k + radial_thickness_k + void_k.  The void between two sheets is
    `pitch - thickness`, except for the gaps k in `pressed_gaps` (a (first, last) pair, both included),
    where it is multiplied by 1 - w and w is a raised-cosine bump of total width `pressed_width` around
    `pressed_theta`: at its centre w = 1 and the sheets touch.  `radial_thickness_k` is `thickness`
    along the normal of the ring, hence `thickness * sqrt(1 + (recto_k' / recto_k)**2)` along the radius;
    on winding `join_winding` it is doubled over `join_width` rad around `join_theta` (cosine flanks of
    0.2 rad *outside* that width, so the plateau really is `join_width` long).  Pass None to switch the
    pressed zone (`pressed_gaps`) or the join (`join_winding`) off.

    Returns {"volume": uint8 (n_z, H, W), "center_xy", "surfaces", "recto", "radial_thickness",
    "sheet_mass", "pressed_theta", "join_theta", "px_mm"}.  H = W is odd and the centre is the middle pixel.
    A pixel is `air_grey + (sheet_grey - air_grey)` times the length of [r - 0.5, r + 0.5] covered by
    sheets, with Gaussian noise of standard deviation `noise` (grey levels) added if asked, rounded and
    clipped to uint8.  `surfaces[k]` is (x, y, z, valid), each (n_z, n_cols): rows are the planes z = 0 ..
    n_z - 1, columns the angles 2*pi*j/n_cols from +x towards +y (the ring is not closed by a repeated
    column), x, y the recto of winding k, valid everywhere.  `recto(k, theta)` and
    `radial_thickness(k, theta)` accept a scalar or an array of angles.
    """
    if n_windings < 1 or n_z < 1 or n_cols < 3:
        raise ValueError("a phantom needs at least one winding, one plane and three columns")
    if not 0 < thickness <= pitch:
        raise ValueError(f"thickness must be in (0, pitch], not {thickness} with pitch {pitch}")
    if join_winding is not None and not 0 <= join_winding < n_windings:
        raise ValueError(f"join_winding {join_winding} is not one of the {n_windings} windings")
    if pressed_gaps is not None:
        first, last = pressed_gaps
        if first > last:
            raise ValueError(f"pressed_gaps {pressed_gaps} must be a (first, last) pair")

    n_nodes = n_cols * max(1, -(-_TABLE_NODES // n_cols))      # every column angle is a node of the tables
    nodes = _TWO_PI * np.arange(n_nodes) / n_nodes
    step = _TWO_PI / n_nodes
    bump = _pressed_bump(nodes, pressed_theta, pressed_width)
    recto_tab = np.empty((n_windings, n_nodes))
    thick_tab = np.empty((n_windings, n_nodes))
    recto_tab[0] = r0
    for k in range(n_windings):
        slope = (np.roll(recto_tab[k], -1) - np.roll(recto_tab[k], 1)) / (2 * step)
        v = _join_plateau(nodes, join_theta, join_width) if k == join_winding else 0.0
        thick_tab[k] = thickness * (1.0 + v) * np.sqrt(1.0 + (slope / recto_tab[k]) ** 2)
        if k + 1 < n_windings:
            pressed = pressed_gaps is not None and pressed_gaps[0] <= k <= pressed_gaps[1]
            void = (pitch - thickness) * ((1.0 - bump) if pressed else 1.0)
            recto_tab[k + 1] = recto_tab[k] + thick_tab[k] + void

    def _row(k: int) -> int:
        if not 0 <= k < n_windings:
            raise IndexError(f"winding {k} is not in 0..{n_windings - 1}")
        return k

    def recto(k, theta):
        return np.interp(theta, nodes, recto_tab[_row(k)], period=_TWO_PI)

    def radial_thickness(k, theta):
        return np.interp(theta, nodes, thick_tab[_row(k)], period=_TWO_PI)

    half = int(np.ceil((recto_tab[-1] + thick_tab[-1]).max())) + 10
    size = 2 * half + 1
    center_xy = (float(half), float(half))
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - center_xy[0], yy - center_xy[1]
    rho, theta = np.hypot(dx, dy), np.arctan2(dy, dx) % _TWO_PI
    covered = np.zeros(rho.shape)
    for k in range(n_windings):
        lo = np.interp(theta, nodes, recto_tab[k], period=_TWO_PI)
        hi = lo + np.interp(theta, nodes, thick_tab[k], period=_TWO_PI)
        covered += np.clip(np.minimum(rho + 0.5, hi) - np.maximum(rho - 0.5, lo), 0.0, None)
    plane = air_grey + (sheet_grey - air_grey) * covered
    volume = np.broadcast_to(plane, (n_z, size, size))
    if noise > 0:
        volume = volume + noise * np.random.default_rng(seed).standard_normal(volume.shape)
    volume = np.clip(np.rint(volume), 0, 255).astype(np.uint8)

    column = np.arange(n_cols) * (n_nodes // n_cols)
    angle = nodes[column]
    surfaces = []
    for k in range(n_windings):
        ring = recto_tab[k][column]
        x = np.tile(center_xy[0] + ring * np.cos(angle), (n_z, 1))
        y = np.tile(center_xy[1] + ring * np.sin(angle), (n_z, 1))
        z = np.repeat(np.arange(n_z, dtype=float)[:, None], n_cols, axis=1)
        surfaces.append((x, y, z, np.ones((n_z, n_cols), dtype=bool)))
    return {
        "volume": volume,
        "center_xy": center_xy,
        "surfaces": surfaces,
        "recto": recto,
        "radial_thickness": radial_thickness,
        "sheet_mass": float((sheet_grey - air_grey) * thickness),
        "pressed_theta": float(pressed_theta),
        "join_theta": float(join_theta),
        "px_mm": float(px_mm),
    }
