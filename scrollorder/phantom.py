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
