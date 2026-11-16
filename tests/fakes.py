"""Test-only fake of the public data bucket (anonymous HTTPS, S3 listing API).

No test may touch the network: everything that reads the server goes through
``FakeBucket`` handed to ``Remote(opener=...)``.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

S3_NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
# Real S3 tokens are opaque and contain characters that must be percent-encoded.
_TOKEN_PREFIX = "tok+/="


def s3_list_xml(prefixes, keys, token=None, *, request_prefix: str = "") -> bytes:
    """One page of a ListObjectsV2 answer.

    prefixes: common prefixes; keys: (key, size) pairs; token: the NextContinuationToken
    (the page is then truncated). Like real S3, the answer starts with a top-level
    <Prefix> (the request prefix), which must not be mistaken for CommonPrefixes/Prefix.
    """
    prefixes, keys = list(prefixes), list(keys)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<ListBucketResult xmlns="{S3_NAMESPACE}">',
        "<Name>fake-bucket</Name>",
        f"<Prefix>{escape(request_prefix)}</Prefix>",
        f"<KeyCount>{len(prefixes) + len(keys)}</KeyCount>",
        f"<IsTruncated>{'true' if token is not None else 'false'}</IsTruncated>",
    ]
    if token is not None:
        parts.append(f"<NextContinuationToken>{escape(token)}</NextContinuationToken>")
    for key, size in keys:
        parts.append(f"<Contents><Key>{escape(key)}</Key><Size>{int(size)}</Size></Contents>")
    for prefix in prefixes:
        parts.append(f"<CommonPrefixes><Prefix>{escape(prefix)}</Prefix></CommonPrefixes>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode("utf-8")


def _not_found(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 404, "Not Found", None, None)


class FakeBucket:
    """Serves the files under ``root`` like the public bucket; call it with a URL, get bytes.

    ``calls`` counts the files actually served (a 404 is not a file read, a listing is not
    either). ``delete_chunk`` reads the metadata straight from ``root`` and never touches it.
    """

    def __init__(self, root, page_size: int = 1000):
        self.root = Path(root)
        self.page_size = page_size
        self.calls = 0
        self._lock = threading.Lock()

    # -- serving ---------------------------------------------------------------------------

    def __call__(self, url: str) -> bytes:
        parts = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        if "list-type" in query:
            return self._list(url, query)
        key = urllib.parse.unquote(parts.path.lstrip("/"))
        path = self._path(key)
        if path is None or not path.is_file():
            raise _not_found(url)
        data = path.read_bytes()
        with self._lock:
            self.calls += 1
        return data

    def _path(self, key: str):
        names = key.split("/")
        if not key or any(n in ("", ".", "..") for n in names):
            return None
        return self.root.joinpath(*names)

    def _list(self, url: str, query) -> bytes:
        prefix = query.get("prefix", [""])[0]
        delimiter = query.get("delimiter", [""])[0]
        token = query.get("continuation-token", [None])[0]
        entries, seen = [], set()           # (name, size) ; size None for a common prefix
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self.root).as_posix()
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if delimiter and delimiter in rest:
                common = prefix + rest.split(delimiter, 1)[0] + delimiter
                if common not in seen:
                    seen.add(common)
                    entries.append((common, None))
            else:
                entries.append((key, path.stat().st_size))
        entries.sort(key=lambda e: e[0])
        offset = 0
        if token is not None:
            if not token.startswith(_TOKEN_PREFIX) or not token[len(_TOKEN_PREFIX):].isdigit():
                raise urllib.error.HTTPError(url, 400, "Invalid continuation token", None, None)
            offset = int(token[len(_TOKEN_PREFIX):])
        page = entries[offset:offset + self.page_size]
        more = offset + self.page_size < len(entries)
        return s3_list_xml(
            [name for name, size in page if size is None],
            [(name, size) for name, size in page if size is not None],
            f"{_TOKEN_PREFIX}{offset + self.page_size}" if more else None,
            request_prefix=prefix,
        )

    # -- building the bucket ---------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        if path is None:
            raise ValueError(f"invalid key {key!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete_chunk(self, array_key: str, idx) -> None:
        """Remove one chunk file of a zarr array (a masked zone): raises if it is not there."""
        from scrollorder.remote import parse_zarr_meta

        base = self.root / array_key
        zarray, zarr_json = base / ".zarray", base / "zarr.json"
        meta = parse_zarr_meta(
            array_key,
            zarray=zarray.read_bytes() if zarray.is_file() else None,
            zarr_json=zarr_json.read_bytes() if zarr_json.is_file() else None,
        )
        (base / meta.chunk_key(tuple(idx))).unlink()


# --- a small sample laid out like the real server ----------------------------------------------

def make_fake_sample(root, sample: str, n_segments: int, with_surface_volume: bool = False) -> FakeBucket:
    """A bucket holding one tiny sample, laid out like the real ``<sample>/...`` tree.

    - ``{sample}/volumes/v.zarr``: OME-Zarr (format 3), one level ``0``, shape (13, 64, 64), uint8;
    - ``{sample}/segments/s-w00{i}_1/``: ``x.tif y.tif z.tif meta.json`` straight in the folder (the real
      server nests them in ``mesh/<id>.tifxyz/``; tests add that layout with ``bucket.put``), and
      ``surface-volumes/sv.zarr/zarr.json`` when ``with_surface_volume``;
    - ``{sample}/representations/predictions/lasagna/l.zarr``: a format 3 array (2 channels), metadata only.
    """
    import zarr

    from tests.helpers import cylinder_grid, write_tifxyz

    root = Path(root)
    volume = zarr.create_group(
        store=str(root / sample / "volumes" / "v.zarr"), zarr_format=3,
        attributes={"ome": {"version": "0.5", "multiscales": [{
            "name": "v",
            "axes": [{"name": a, "type": "space"} for a in "zyx"],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
        }]}})
    level0 = volume.create_array("0", shape=(13, 64, 64), chunks=(8, 32, 32), dtype="uint8", fill_value=0)
    level0[:] = np.random.default_rng(0).integers(0, 255, (13, 64, 64), dtype=np.uint8)
    zarr.create_array(
        store=str(root / sample / "representations" / "predictions" / "lasagna" / "l.zarr"),
        shape=(2, 13, 16, 16), chunks=(1, 13, 16, 16), dtype="uint8", fill_value=0, zarr_format=3,
        attributes={"channels": ["grad_mag", "cos"]})
    x, y, z = cylinder_grid(n_rows=4, n_cols=8)
    for i in range(n_segments):
        folder = root / sample / "segments" / f"s-w{i:03d}_1"
        write_tifxyz(folder, x, y, z)
        if with_surface_volume:
            (folder / "surface-volumes" / "sv.zarr").mkdir(parents=True)
            (folder / "surface-volumes" / "sv.zarr" / "zarr.json").write_text(json.dumps(
                {"zarr_format": 3, "node_type": "group", "attributes": {}}))
    return FakeBucket(root)


def write_fake_config(tmp, sample: str, n_segments: int) -> Path:
    """The ``config/<sample>.json`` that goes with ``make_fake_sample``; returns its path."""
    cfg = {
        "sample": sample,
        "voxel_mm": 0.002399,
        "volume": {"group": f"{sample}/volumes/v.zarr", "levels": {"0": "0"}, "axes": "zyx"},
        "pyramid": "mean",
        "segments": [{"name": f"w{i:03d}", "prefix": f"{sample}/segments/s-w{i:03d}_1/"} for i in range(n_segments)],
        "tifxyz_to_l0": 1.0,
        "lasagna": {"level": 0, "grad_scale": 1000.0, "channels": {
            "grad_mag": {"key": f"{sample}/representations/predictions/lasagna/l.zarr", "index": 0},
            "cos": {"key": f"{sample}/representations/predictions/lasagna/l.zarr", "index": 1}}},
        "recto_offset_px": 0.0,
        "mass_from_l0": False,
        "overlay_ok": None,
    }
    path = Path(tmp) / f"{sample}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


# --- a two-level volume ------------------------------------------------------------------------

def _next_level(level0: np.ndarray, pyramid: str) -> np.ndarray:
    """Level 1 of a volume, with the shape rule of the real server: ceil(shape / 2) on every axis.

    "stride" keeps every second voxel from index 0; "mean" averages each 2x2x2 block over the voxels
    it really has (an edge block of an odd axis is smaller), rounded for an integer dtype.
    """
    if pyramid == "stride":
        return level0[::2, ::2, ::2]
    sums, counts = level0.astype(np.float64), np.ones(level0.shape)
    for axis, n in enumerate(level0.shape):
        starts = np.arange(0, n, 2)
        sums = np.add.reduceat(sums, starts, axis=axis)
        counts = np.add.reduceat(counts, starts, axis=axis)
    mean = sums / counts
    if np.issubdtype(level0.dtype, np.integer):
        mean = np.rint(mean)
    return mean.astype(level0.dtype)


def make_fake_volume(root, data=None, dtype="uint8", pyramid="mean"):
    """A two-level zarr v2 volume behind a ``FakeBucket``, returned as the ``Volume`` that reads it.

    ``root/bucket`` holds the group ``vol.zarr`` (arrays ``0`` and ``1``, axes z, y, x) and ``root/cache``
    is the ``Remote`` cache; ``volume.remote.opener`` is the ``FakeBucket``. ``data`` is level 0, cast
    to ``dtype`` (default: seed-0 random integers in [0, 255) of shape (13, 64, 64)). Level 1 is
    ``ceil(shape / 2)`` big: block means for ``"mean"``, one voxel out of two for ``"stride"``.
    """
    import zarr

    from scrollorder.remote import Remote
    from scrollorder.volume import Volume

    if pyramid not in ("mean", "stride"):
        raise ValueError(f"unknown pyramid {pyramid!r}")
    root = Path(root)
    if data is None:
        data = np.random.default_rng(0).integers(0, 255, (13, 64, 64), dtype=np.uint8)
    level0 = np.asarray(data).astype(dtype)
    level1 = _next_level(level0, pyramid)
    group = zarr.create_group(
        store=str(root / "bucket" / "vol.zarr"), zarr_format=2,
        attributes={"multiscales": [{
            "version": "0.4", "name": "vol",
            "axes": [{"name": a, "type": "space"} for a in "zyx"],
            "datasets": [{"path": "0"}, {"path": "1"}],
        }]})
    for name, values, chunks in (("0", level0, (8, 32, 32)), ("1", level1, (4, 16, 16))):
        array = group.create_array(name, shape=values.shape, chunks=chunks, dtype=values.dtype, fill_value=0)
        array[:] = values
    remote = Remote(base_url="https://h", cache_dir=root / "cache", opener=FakeBucket(root / "bucket"))
    return Volume(remote=remote, group="vol.zarr", levels={0: "0", 1: "1"}, voxel_mm=0.002399, pyramid=pyramid)
