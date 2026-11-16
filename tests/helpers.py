"""Test-only utilities."""
import json
from pathlib import Path

import numpy as np
import tifffile


def write_tifxyz(directory, x, y, z, scale=(0.05, 0.05), mask=None):
    """Write a TIFXYZ folder as specified in villa/lasagna/tifxyz_format.md."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in (("x", x), ("y", y), ("z", z)):
        tifffile.imwrite(d / f"{name}.tif", np.asarray(arr, dtype=np.float32))
    if mask is not None:
        tifffile.imwrite(d / "mask.tif", np.asarray(mask, dtype=np.uint8))
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": list(scale)}))
    return d


def cylinder_grid(center_xy=(120.0, 90.0), radius=40.0, z_range=(10.0, 60.0), n_rows=26, n_cols=181, arc=2 * np.pi):
    theta = np.linspace(0.0, arc, n_cols)
    zs = np.linspace(*z_range, n_rows)
    x = np.broadcast_to(center_xy[0] + radius * np.cos(theta), (n_rows, n_cols)).copy()
    y = np.broadcast_to(center_xy[1] + radius * np.sin(theta), (n_rows, n_cols)).copy()
    z = np.broadcast_to(zs[:, None], (n_rows, n_cols)).copy()
    return x, y, z
