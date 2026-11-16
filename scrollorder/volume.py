"""Reading the CT volume (an OME-Zarr pyramid) and moving between its levels.

Level ``n`` is the level-0 volume reduced ``2**n`` times along each axis. How a level coordinate
relates to a level-0 coordinate depends on how the pyramid was built (``DataConfig.pyramid``):

- ``"mean"``: a level voxel is the mean of the ``2**n`` level-0 voxels it covers, so its centre is
  the middle of that block: ``c_level = (c0 + 0.5) / 2**n - 0.5``;
- ``"stride"``: a level voxel is the level-0 voxel at ``2**n * c_level``: ``c_level = c0 / 2**n``.

Coordinates are voxel indices (the centre of voxel ``i`` is ``i``); ``z``, ``y``, ``x`` are the axes
of the arrays. Only reading goes through the network, via ``scrollorder.remote``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from scrollorder.dataconf import PYRAMIDS, DataConfig
from scrollorder.remote import Remote, read_zarr_region, zarr_meta


def _check_pyramid(pyramid: str) -> None:
    if pyramid not in PYRAMIDS:
        raise ValueError(f'pyramid must be "mean" or "stride", not {pyramid!r}')


def l0_to_level(c, level: int, pyramid: str):
    """Coordinate ``c`` (level 0, scalar or array) expressed on the grid of ``level``."""
    _check_pyramid(pyramid)
    scale = 2 ** level
    if pyramid == "mean":
        return (c + 0.5) / scale - 0.5
    return c / scale


def level_to_l0(c, level: int, pyramid: str):
    """Coordinate ``c`` of ``level`` (scalar or array) expressed on the level-0 grid: inverse of ``l0_to_level``."""
    _check_pyramid(pyramid)
    scale = 2 ** level
    if pyramid == "mean":
        return (c + 0.5) * scale - 0.5
    return c * scale


def nearest_plane(z_l0: float, level: int, pyramid: str) -> int:
    """Index of the plane of ``level`` closest to level-0 coordinate ``z_l0`` (a tie goes up)."""
    return int(np.floor(l0_to_level(z_l0, level, pyramid) + 0.5))


def px_mm(voxel_mm: float, level: int) -> float:
    """Edge of a voxel of ``level``, given the edge of a level-0 voxel."""
    return voxel_mm * 2 ** level


@dataclass(eq=False)
class SliceImage:
    """One plane of one level, cropped to a window.

    ``data`` is float32 (rows are y, columns are x); ``origin_xy`` is the (x, y) index, on the grid of
    ``level``, of ``data[0, 0]``; ``z_l0`` is the level-0 coordinate of the plane; ``px_mm`` the pixel edge.
    """

    data: np.ndarray
    origin_xy: tuple[int, int]
    level: int
    plane: int
    z_l0: float
    px_mm: float


@dataclass
class Volume:
    """The pyramid of one scan: ``levels`` maps a level to the path of its array relative to ``group``.

    ``group`` is the bucket key of the zarr group. Array shapes and dtypes are read once per level
    and kept in the instance. The arrays are indexed ``[z, y, x]``.
    """

    remote: Remote
    group: str
    levels: dict[int, str]
    voxel_mm: float                      # edge of a level-0 voxel
    pyramid: str                         # "mean" or "stride"
    _info: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        _check_pyramid(self.pyramid)
        if 0 not in self.levels:
            raise ValueError(f"level 0 is missing from the levels {sorted(self.levels)}")

    @classmethod
    def from_config(cls, cfg: DataConfig, remote: Remote) -> Volume:
        """The volume described by ``cfg.volume`` (the JSON level keys are strings: they become integers)."""
        axes = cfg.volume.get("axes", "zyx")
        if axes != "zyx":
            raise ValueError(f'volume axes must be "zyx", not {axes!r}')
        levels = {int(level): path for level, path in cfg.volume["levels"].items()}
        return cls(remote=remote, group=cfg.volume["group"], levels=levels,
                   voxel_mm=cfg.voxel_mm, pyramid=cfg.pyramid)

    # -- metadata --------------------------------------------------------------------------

    def _array_key(self, level: int) -> str:
        if level not in self.levels:
            raise ValueError(f"no level {level} in this volume (levels: {sorted(self.levels)})")
        return f"{self.group.rstrip('/')}/{self.levels[level]}"

    def _array_info(self, level: int) -> tuple[tuple[int, int, int], np.dtype]:
        if level not in self._info:
            meta = zarr_meta(self.remote, self._array_key(level))
            if len(meta.shape) != 3:
                raise ValueError(f"level {level} has {len(meta.shape)} axes, z, y, x were expected")
            self._info[level] = (tuple(meta.shape), meta.dtype)
        return self._info[level]

    def shape(self, level: int) -> tuple[int, int, int]:
        """(z, y, x) size of the array of ``level``."""
        return self._array_info(level)[0]

    def dtype_max(self) -> float:
        """Largest value of the stored type (255.0 for uint8); 1.0 for a floating-point volume."""
        dtype = self._array_info(0)[1]
        if np.issubdtype(dtype, np.integer):
            return float(np.iinfo(dtype).max)
        return 1.0

    # -- reading ---------------------------------------------------------------------------

    def block(self, level: int, z: slice, y: slice, x: slice) -> np.ndarray:
        """Voxels of ``level`` over the three slices, as float32; slices are clipped like numpy's."""
        region = tuple(slice(*s.indices(n)) for s, n in zip((z, y, x), self.shape(level)))
        return read_zarr_region(self.remote, self._array_key(level), region).astype(np.float32)

    def read_slice(self, level: int, plane: int, bbox_xy_level: tuple[float, float, float, float],
                   margin_px: int, from_l0: bool = False) -> SliceImage:
        """Plane ``plane`` of ``level`` around a bounding box ``(x_min, y_min, x_max, y_max)`` in pixels of ``level``.

        The window holds the columns from ``floor(x_min) - margin_px`` to ``ceil(x_max) + margin_px``
        included, likewise for the rows, cropped to the array; ``origin_xy`` is its first pixel.
        Raises ``IndexError`` if ``plane`` is outside the array and ``ValueError`` if the window does
        not meet it.

        With ``from_l0`` the pixels are not read from ``level`` but computed from level 0: each one is
        the mean of the ``2**level`` planes ``[plane * 2**level, (plane + 1) * 2**level)`` and of its
        ``2**level x 2**level`` block of columns and rows (the blocks are aligned on the grid of
        ``level``). Where level 0 ends before the block does, only the voxels present are averaged.
        The planes and the whole window of level 0 are read in one go, in their stored dtype: the
        memory is ``2**(3 * level)`` times the window.

        Known limitation: with ``pyramid="stride"`` the averaged planes are centred like a ``"mean"``
        pyramid whereas ``z_l0`` follows the ``"stride"`` formula, so they differ by up to
        ``(2**level - 1) / 2`` level-0 voxels along z.
        """
        n_planes, n_rows, n_cols = self.shape(level)
        if not 0 <= plane < n_planes:
            raise IndexError(f"plane {plane} is outside level {level} (0 to {n_planes - 1})")
        x_min, y_min, x_max, y_max = bbox_xy_level
        x0, x1 = max(math.floor(x_min) - margin_px, 0), min(math.ceil(x_max) + margin_px + 1, n_cols)
        y0, y1 = max(math.floor(y_min) - margin_px, 0), min(math.ceil(y_max) + margin_px + 1, n_rows)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"window x {x0}..{x1 - 1}, y {y0}..{y1 - 1} does not meet level {level} "
                             f"({n_cols} columns, {n_rows} rows)")
        if from_l0:
            data = self._mean_of_level0(level, plane, (y0, y1), (x0, x1))
        else:
            data = self.block(level, slice(plane, plane + 1), slice(y0, y1), slice(x0, x1))[0]
        return SliceImage(data=data, origin_xy=(x0, y0), level=level, plane=plane,
                          z_l0=float(level_to_l0(plane, level, self.pyramid)),
                          px_mm=px_mm(self.voxel_mm, level))

    def _mean_of_level0(self, level: int, plane: int, rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
        """Float32 ``(rows, cols)`` window of ``level`` computed from level 0 (see ``read_slice``)."""
        f = 2 ** level
        (y0, y1), (x0, x1) = rows, cols
        n_planes0, n_rows0, n_cols0 = self.shape(0)
        z_lo, z_hi = plane * f, min((plane + 1) * f, n_planes0)
        y_lo, y_hi = y0 * f, min(y1 * f, n_rows0)
        x_lo, x_hi = x0 * f, min(x1 * f, n_cols0)
        if z_hi <= z_lo or y_hi <= y_lo or x_hi <= x_lo:
            raise ValueError(f"the window of level {level} lies outside level 0")
        raw = read_zarr_region(self.remote, self._array_key(0),
                               (slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi)))
        plane_sum = raw.sum(axis=0, dtype=np.float64)
        row_starts = np.arange(0, plane_sum.shape[0], f)
        col_starts = np.arange(0, plane_sum.shape[1], f)
        if len(row_starts) != y1 - y0 or len(col_starts) != x1 - x0:
            raise ValueError(f"level {level} is larger than level 0 reduced {f} times")
        sums = np.add.reduceat(np.add.reduceat(plane_sum, row_starts, axis=0), col_starts, axis=1)
        rows_present = np.diff(np.append(row_starts, plane_sum.shape[0]))      # voxels per block, edge blocks included
        cols_present = np.diff(np.append(col_starts, plane_sum.shape[1]))
        counts = (z_hi - z_lo) * np.outer(rows_present, cols_present)
        return (sums / counts).astype(np.float32)
