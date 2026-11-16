"""What a sample holds on the public server, and the download of its traced surfaces.

The server keeps thousands of keys inside some ``*.zarr/`` folders (``surface-volumes/``,
``representations/predictions/``): the inventory therefore walks folder by folder with delimiter
listings and never lists inside a zarr folder. A folder is taken for a zarr when its name ends in
``.zarr`` (not listed at all) or when it directly holds zarr metadata (its one listing is what shows
it); either way it is reported, not opened. The only requests made inside a zarr are the single
metadata files read by ``read_multiscales`` and ``zarr_meta``.

Nothing here knows about a particular sample: names come from the arguments or the configuration.
"""
from __future__ import annotations

import json
import math
import posixpath
from dataclasses import dataclass, field
from pathlib import Path

from .dataconf import DataConfig
from .remote import Remote, read_multiscales, zarr_meta

ZARR_MARKERS = frozenset({"zarr.json", ".zgroup", ".zarray", ".zattrs"})
SURFACE_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")      # a tifxyz folder cannot be used without them
OPTIONAL_SURFACE_FILES = ("mask.tif",)
SHALLOW_DEPTH = 2                                             # listing levels for folders we only glance at
# A zarr with absent or malformed metadata is reported and the run goes on. Anything else (a refused
# request, a full cache, ...) is not about that zarr and stops the run.
_BAD_METADATA = (FileNotFoundError, ValueError, KeyError, IndexError, TypeError)


@dataclass
class _Tree:
    """What a walk found: file keys, folder prefixes (with their trailing slash) and zarr folder keys."""

    files: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    zarrs: list[str] = field(default_factory=list)            # no trailing slash

    def update(self, other: _Tree) -> None:
        self.files += other.files
        self.folders += other.folders
        self.zarrs += other.zarrs


def _is_zarr_name(folder: str) -> bool:
    return folder.rstrip("/").endswith(".zarr")


def _walk(remote: Remote, root: str, *, max_depth: int | None = None, skip: frozenset[str] = frozenset()) -> _Tree:
    """Everything under ``root``, folder by folder, never listing inside a zarr.

    ``max_depth`` caps the number of listing levels (``None``: no cap); folders found one level
    below the cap are still reported by name. Folders in ``skip`` are reported but not entered.
    """
    tree = _Tree()
    stack = [(root, 1)]
    while stack:
        prefix, depth = stack.pop()
        folders, keys = remote.list(prefix)
        if any(key.rpartition("/")[2] in ZARR_MARKERS for key, _ in keys):
            tree.zarrs.append(prefix.rstrip("/"))             # a zarr without the suffix: its chunks are not ours to list
            continue
        tree.files += [key for key, _ in keys]
        for folder in folders:
            if _is_zarr_name(folder):
                tree.zarrs.append(folder.rstrip("/"))
                continue
            tree.folders.append(folder)
            if folder not in skip and (max_depth is None or depth < max_depth):
                stack.append((folder, depth + 1))
    return tree


# --- zarr descriptions -----------------------------------------------------------------------

def _non_finite_text(value: float) -> str:
    """The JSON constant that names ``value`` (``NaN``, ``Infinity``, ``-Infinity``), which is not a valid JSON number."""
    return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"


def _json_number(value: float):
    """``value`` as a JSON number, or as text when it is not finite."""
    return float(value) if math.isfinite(value) else _non_finite_text(value)


def _parse_float(text: str):
    value = float(text)                                       # 1e999 reads as infinity
    return value if math.isfinite(value) else _non_finite_text(value)


def _load_metadata(data: bytes):
    """``json.loads`` for zarr metadata that will be copied into the inventory.

    Writers such as numpy or python emit bare ``NaN`` / ``Infinity`` / ``-Infinity`` (invalid JSON
    that ``json.loads`` accepts as floats). Anywhere in the document, nested or in a list, they
    (and numbers too big for a float) become the text of that constant, so the inventory is strict JSON.
    """
    return json.loads(data, parse_constant=str, parse_float=_parse_float)


def _node_metadata(remote: Remote, key: str) -> tuple[dict, list | None]:
    """(attributes, dimension_names) of the zarr node at ``key``, from its metadata file.

    ``read_multiscales`` and ``zarr_meta`` have just fetched ``zarr.json`` (format 3) or ``.zattrs``
    (format 2 groups); a format 2 array's ``.zattrs`` costs one request, a 404 meaning "none".
    """
    v3 = remote.local_path(f"{key}/zarr.json")
    if v3.is_file():
        doc = _load_metadata(v3.read_bytes())
        return doc.get("attributes") or {}, doc.get("dimension_names")
    v2 = remote.fetch(f"{key}/.zattrs")
    return (_load_metadata(v2.read_bytes()) if v2 is not None else {}), None


def _describe_array(remote: Remote, key: str, path: str, *, with_attributes: bool) -> dict:
    meta = zarr_meta(remote, key)
    level = {"path": path, "key": key, "zarr_format": meta.zarr_format, "shape": list(meta.shape),
             "chunks": list(meta.chunks), "dtype": str(meta.dtype), "fill_value": _json_number(meta.fill_value)}
    attributes, dimension_names = _node_metadata(remote, key)
    if dimension_names:
        level["dimension_names"] = dimension_names
    if with_attributes and attributes:
        level["attributes"] = attributes
    return level


def _read_zarr(remote: Remote, key: str) -> tuple[list[dict], dict]:
    """(levels, root attributes) of a multiscales group, or of a lone array (one level with path "")."""
    try:
        paths = read_multiscales(remote, key)
    except _BAD_METADATA as no_scales:
        try:
            levels = [_describe_array(remote, key, "", with_attributes=False)]
        except _BAD_METADATA as no_array:
            raise ValueError(f"{no_scales}; {no_array}") from no_array
    else:
        levels = []
        for path in paths:
            try:
                levels.append(_describe_array(remote, f"{key}/{path}", path, with_attributes=True))
            except _BAD_METADATA as exc:                      # one broken level does not hide the others
                levels.append({"path": path, "key": f"{key}/{path}", "error": str(exc)})
    return levels, _node_metadata(remote, key)[0]


def _describe_zarr(remote: Remote, key: str) -> dict:
    entry = {"group": key, "name": key.rpartition("/")[2], "levels": []}
    try:
        entry["levels"], entry["attributes"] = _read_zarr(remote, key)
    except _BAD_METADATA as exc:
        entry["error"] = str(exc)
    return entry


# --- inventory -------------------------------------------------------------------------------

def inventory(remote: Remote, sample: str) -> dict:
    """The tree of ``sample`` on the server, as plain JSON data.

    - ``volumes``: every zarr under ``volumes/`` with its levels (shape, chunks, dtype);
    - ``segments``: every folder of ``segments/`` with its files relative to the folder, the zarr
      folders it holds (reported, never listed) and the folders that hold an ``x.tif``;
    - ``predictions``: every zarr under ``representations/predictions/`` (any depth), described like a volume;
    - ``kollesis_keys``: every listed key or folder whose path, below the sample, contains ``kolles``;
    - ``top_level``: the folders directly under the sample.

    Walks ``volumes/``, ``segments/`` and ``representations/predictions/`` down to their last folder,
    and the other top-level folders two levels deep only.
    """
    root = f"{sample.strip('/')}/"
    top_folders, top_files = remote.list(root)
    if not top_folders and not top_files:
        raise FileNotFoundError(f"nothing under {root!r} on the server: check the sample name")
    listed = _Tree(files=[key for key, _ in top_files], folders=list(top_folders))

    volumes = _walk(remote, f"{root}volumes/")
    predictions_root = f"{root}representations/predictions/"
    predictions = _walk(remote, predictions_root)
    listed.update(volumes)
    listed.update(predictions)
    for folder in top_folders:
        if _is_zarr_name(folder):
            listed.zarrs.append(folder.rstrip("/"))
        elif folder not in (f"{root}volumes/", f"{root}segments/"):
            listed.update(_walk(remote, folder, max_depth=SHALLOW_DEPTH, skip=frozenset({predictions_root})))

    segments = []
    segment_folders, _ = remote.list(f"{root}segments/")
    for folder in sorted(f for f in segment_folders if not _is_zarr_name(f)):
        tree = _walk(remote, folder)
        listed.update(tree)
        files = sorted(key[len(folder):] for key in tree.files)
        segments.append({
            "name": folder.rstrip("/").rpartition("/")[2],
            "prefix": folder,
            "files": files,
            "zarr_folders": sorted(key[len(folder):] for key in tree.zarrs),
            "tifxyz_folders": sorted({posixpath.dirname(f) for f in files if posixpath.basename(f) == "x.tif"}),
        })

    names = listed.files + listed.folders + listed.zarrs
    return {
        "sample": sample,
        "top_level": sorted(folder[len(root):].rstrip("/") for folder in top_folders),
        "volumes": [_describe_zarr(remote, key) for key in sorted(volumes.zarrs)],
        "segments": segments,
        "predictions": [_describe_zarr(remote, key) for key in sorted(predictions.zarrs)],
        "kollesis_keys": sorted({n for n in names if "kolles" in n[len(root):].lower()}),
    }


def entries_in_error(inv: dict) -> list[tuple[str, str]]:
    """(key, message) of every volume, prediction or level of ``inv`` whose metadata could not be read."""
    found = []
    for entry in inv["volumes"] + inv["predictions"]:
        if "error" in entry:
            found.append((entry["group"], entry["error"]))
        found += [(level["key"], level["error"]) for level in entry["levels"] if "error" in level]
    return found


# --- surfaces --------------------------------------------------------------------------------

def fetch_segments(remote: Remote, cfg: DataConfig) -> list[Path]:
    """Download the tifxyz folder of every segment of ``cfg``; return the local folders in ``cfg.segments`` order.

    ``x.tif``, ``y.tif``, ``z.tif`` and ``meta.json`` are mandatory (a missing one raises
    ``FileNotFoundError`` naming the segment); ``mask.tif`` is taken when the server has it.
    """
    folders = []
    for segment in cfg.segments:
        base = segment["prefix"].rstrip("/")
        for name in SURFACE_FILES:
            if remote.fetch(f"{base}/{name}") is None:
                raise FileNotFoundError(f"segment {segment['name']!r}: {name} is not on the server ({base}/)")
        for name in OPTIONAL_SURFACE_FILES:
            remote.fetch(f"{base}/{name}")
        folders.append(remote.local_path(base))
    return folders
