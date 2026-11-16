"""Anonymous HTTPS access to the public data bucket, with a local cache zarr can read from.

Everything goes through ``Remote``: files are downloaded once into ``cache_dir`` (same layout as
the bucket), the cache is capped by ``budget_bytes``, and zarr arrays are read from that local
copy. The network is reached only through ``Remote.opener`` (``url -> bytes``), so tests inject
a fake server and never touch the network.

``zarr`` is an optional dependency (extra ``real``): it is imported only by ``read_zarr_region``.
"""
from __future__ import annotations

import http.client
import itertools
import json
import operator
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scrollorder.thresholds import CACHE_BUDGET_GB

BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HTTP_TIMEOUT_S = 60.0


def default_cache_dir() -> Path:
    """``$SCROLLORDER_CACHE`` if set, else ``~/.cache/scrollorder`` (always outside the repository)."""
    env = os.environ.get("SCROLLORDER_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "scrollorder"


def http_get(url: str) -> bytes:
    """GET ``url`` anonymously. Raises ``urllib.error.HTTPError`` (404 included) or ``OSError``.

    The plain URL goes through ``urllib`` with its default User-Agent: nothing identifying is sent.
    """
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
        return response.read()


class CacheBudgetExceeded(RuntimeError):
    """Writing one more file would take the cache over its budget: stop and ask."""


@dataclass(eq=False)
class Remote:
    """The bucket and its local cache.

    ``fetch`` retries once per entry of ``retry_waits`` (after sleeping that long) on ``OSError`` and
    ``http.client.HTTPException``, never on a 404. The cache size is counted once by walking
    ``cache_dir`` and then kept up to date by this object: two ``Remote`` objects sharing one
    ``cache_dir`` each enforce the budget against what they know.
    """

    base_url: str = BUCKET_URL
    cache_dir: Path = field(default_factory=default_cache_dir)
    budget_bytes: int = CACHE_BUDGET_GB * 2**30
    opener: Callable[[str], bytes] = http_get
    retry_waits: tuple[float, ...] = (1.0, 4.0, 16.0)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _total: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)

    # -- paths and requests ----------------------------------------------------------------

    def local_path(self, key: str) -> Path:
        """Where ``key`` lives (or would live) in the cache. Refuses keys that leave ``cache_dir``."""
        names = key.split("/")
        if any(n in ("", ".", "..") for n in names):
            raise ValueError(f"invalid key {key!r}")
        return self.cache_dir.joinpath(*names)

    def _url(self, key: str) -> str:
        return f"{self.base_url.rstrip('/')}/{urllib.parse.quote(key, safe='/')}"

    def _get(self, url: str) -> bytes | None:
        """The body of ``url``; ``None`` on a 404. Retries, then raises ``OSError``."""
        last: Exception | None = None
        for attempt in range(len(self.retry_waits) + 1):
            if attempt:
                time.sleep(self.retry_waits[attempt - 1])
            try:
                return self.opener(url)
            except urllib.error.HTTPError as exc:       # before OSError: HTTPError is one
                if exc.code == 404:
                    return None
                last = exc
            except (OSError, http.client.HTTPException) as exc:
                last = exc
        if isinstance(last, OSError):
            raise last
        raise OSError(f"download of {url} failed after {len(self.retry_waits) + 1} attempts: {last!r}") from last

    # -- files -----------------------------------------------------------------------------

    def fetch(self, key: str) -> Path | None:
        """Local path of ``key``, downloading it the first time; ``None`` if the server has no such file."""
        target = self.local_path(key)
        if target.is_file():
            return target
        data = self._get(self._url(key))
        if data is None:
            return None
        with self._lock:                                  # recheck, budget, write and count as one step
            if target.is_file():                          # another thread got there first
                return target
            total = self._known_total()
            if total + len(data) > self.budget_bytes:
                raise CacheBudgetExceeded(
                    f"fetching {key!r} ({len(data)} bytes) would take the cache from {total} bytes "
                    f"over its budget of {self.budget_bytes} bytes ({self.cache_dir})")
            target.parent.mkdir(parents=True, exist_ok=True)
            # unique per writer (process and call): two processes filling one key never share a partial file
            part = target.with_name(f"{target.name}.{os.getpid()}.{secrets.token_hex(4)}.part")
            try:
                part.write_bytes(data)
                os.replace(part, target)
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            self._total = total + len(data)
        return target

    def cache_bytes(self) -> int:
        """Bytes held by the cache (counted on first use, then kept up to date)."""
        with self._lock:
            return self._known_total()

    def _known_total(self) -> int:
        if self._total is None:                           # caller holds self._lock
            total = 0
            if self.cache_dir.is_dir():
                for path in self.cache_dir.rglob("*"):
                    try:
                        if path.is_file():
                            total += path.stat().st_size
                    except OSError:
                        pass
            self._total = total
        return self._total

    # -- listing ---------------------------------------------------------------------------

    def list(self, prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[tuple[str, int]]]:
        """S3 ListObjectsV2 under ``prefix``, all pages: (sub-folders, (key, size)).

        With ``delimiter=None`` there are no sub-folders: every key below ``prefix`` is returned.
        """
        quote = lambda s: urllib.parse.quote(s, safe="")   # noqa: E731
        prefixes: list[str] = []
        keys: list[tuple[str, int]] = []
        token = None
        while True:
            query = ["list-type=2", f"prefix={quote(prefix)}"]
            if delimiter is not None:
                query.append(f"delimiter={quote(delimiter)}")
            if token is not None:
                query.append(f"continuation-token={quote(token)}")
            body = self._get(f"{self.base_url.rstrip('/')}/?{'&'.join(query)}")
            if body is None:
                raise FileNotFoundError(f"no listing at {self.base_url} (404)")
            more, next_token = _parse_listing(body, prefixes, keys)
            if not more:
                return prefixes, keys
            if not next_token or next_token == token:
                raise OSError("truncated listing without a new continuation token")
            token = next_token


def _local_name(tag: str) -> str:
    return tag.rpartition("}")[2]


def _parse_listing(body: bytes, prefixes: list[str], keys: list[tuple[str, int]]) -> tuple[bool, str | None]:
    """Add one page to ``prefixes`` and ``keys``; return (IsTruncated, NextContinuationToken)."""
    try:
        text = body.decode("utf-8-sig")                # strict: other encodings could hide a DTD from the check
    except UnicodeDecodeError as exc:
        raise OSError("listing answer is not UTF-8") from exc
    if "<!DOCTYPE" in text or "<!ENTITY" in text:      # S3 never sends a DTD: refuse entity tricks
        raise OSError("unexpected DTD in a listing answer")
    try:
        root = ET.fromstring(text)                     # the text that was checked is the text that is parsed
    except ET.ParseError as exc:
        raise OSError(f"unreadable listing: {exc}") from exc
    if _local_name(root.tag) != "ListBucketResult":
        raise OSError(f"unexpected listing answer: <{_local_name(root.tag)}>")
    truncated, token = False, None
    for child in root:                  # direct children only: the top-level <Prefix> is not a sub-folder
        name = _local_name(child.tag)
        if name == "CommonPrefixes":
            prefixes.extend(p.text or "" for p in child if _local_name(p.tag) == "Prefix")
        elif name == "Contents":
            fields = {_local_name(e.tag): (e.text or "") for e in child}
            if "Key" not in fields:
                raise OSError("malformed listing: <Contents> without <Key>")
            try:
                size = int(fields.get("Size") or 0)
            except ValueError as exc:
                raise OSError(f"malformed listing: size {fields['Size']!r} of {fields['Key']!r}") from exc
            keys.append((fields["Key"], size))
        elif name == "IsTruncated":
            truncated = (child.text or "").strip().lower() == "true"
        elif name == "NextContinuationToken":
            token = child.text
    return truncated, token


# --- zarr ------------------------------------------------------------------------------------

@dataclass
class ZarrMeta:
    """What the pipeline needs to know about one zarr array.

    ``chunks`` is the shape of one *file*: with zarr v3 sharding it is the shard shape (the inner
    chunks live inside the shard files). ``meta_keys`` are full bucket keys of the metadata files
    (``.zattrs`` may be absent on the server). ``chunk_key_encoding`` (``"v2"`` or ``"default"``)
    and ``separator`` say how ``chunk_key`` spells a chunk's path.
    """

    zarr_format: int
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    fill_value: float
    meta_keys: tuple[str, ...]
    chunk_key_encoding: str = "v2"
    separator: str = "."

    def chunk_key(self, idx: tuple[int, ...]) -> str:
        """Path of chunk ``idx`` (chunk-grid indices) relative to the array."""
        parts = [str(int(i)) for i in idx]
        if self.chunk_key_encoding == "default":
            return self.separator.join(["c", *parts])
        return self.separator.join(parts or ["0"])


def _fill_value(raw) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, (bool, int, float)):
        return float(raw)
    if raw == "NaN":
        return float("nan")
    if raw == "Infinity":
        return float("inf")
    if raw == "-Infinity":
        return float("-inf")
    raise ValueError(f"unsupported fill_value {raw!r}")


def parse_zarr_meta(array_key: str, *, zarray: bytes | None = None, zarr_json: bytes | None = None) -> ZarrMeta:
    """Describe an array from the bytes of its ``.zarray`` (format 2) or ``zarr.json`` (format 3)."""
    key = array_key.rstrip("/")
    if zarray is not None:
        doc = json.loads(zarray)
        return ZarrMeta(
            zarr_format=2,
            shape=tuple(int(n) for n in doc["shape"]),
            chunks=tuple(int(n) for n in doc["chunks"]),
            dtype=np.dtype(doc["dtype"]),
            fill_value=_fill_value(doc.get("fill_value")),
            meta_keys=(f"{key}/.zarray", f"{key}/.zattrs"),
            chunk_key_encoding="v2",
            separator=doc.get("dimension_separator") or ".",
        )
    if zarr_json is None:
        raise ValueError(f"no zarr metadata for {array_key!r}")
    doc = json.loads(zarr_json)
    if doc.get("node_type") != "array":
        raise ValueError(f"{array_key!r} is not a zarr array (node_type={doc.get('node_type')!r})")
    grid = doc.get("chunk_grid") or {}
    if grid.get("name") != "regular":
        raise ValueError(f"unsupported chunk grid {grid.get('name')!r} in {array_key!r}")
    encoding = doc.get("chunk_key_encoding") or {"name": "default"}
    style = encoding.get("name")
    if style not in ("default", "v2"):
        raise ValueError(f"unsupported chunk_key_encoding {style!r} in {array_key!r}")
    separator = (encoding.get("configuration") or {}).get("separator") or ("/" if style == "default" else ".")
    try:
        dtype = np.dtype(doc["data_type"])
    except TypeError as exc:
        raise ValueError(f"unsupported data_type {doc['data_type']!r} in {array_key!r}") from exc
    return ZarrMeta(
        zarr_format=3,
        shape=tuple(int(n) for n in doc["shape"]),
        chunks=tuple(int(n) for n in grid["configuration"]["chunk_shape"]),
        dtype=dtype,
        fill_value=_fill_value(doc.get("fill_value")),
        meta_keys=(f"{key}/zarr.json",),
        chunk_key_encoding=style,
        separator=separator,
    )


def _fetch_first(remote: Remote, keys: list[str]) -> tuple[str, Path] | None:
    """First of ``keys`` that exists: already in the cache, else on the server (no request if cached)."""
    for key in keys:
        path = remote.local_path(key)
        if path.is_file():
            return key, path
    for key in keys:
        path = remote.fetch(key)
        if path is not None:
            return key, path
    return None


def zarr_meta(remote: Remote, array_key: str) -> ZarrMeta:
    """Metadata of the array at ``array_key``: format 2 if ``.zarray`` exists, else format 3 (``zarr.json``)."""
    key = array_key.rstrip("/")
    found = _fetch_first(remote, [f"{key}/.zarray", f"{key}/zarr.json"])
    if found is None:
        raise FileNotFoundError(f"no zarr array at {key!r}")
    name, path = found
    if name.endswith(".zarray"):
        return parse_zarr_meta(key, zarray=path.read_bytes())
    return parse_zarr_meta(key, zarr_json=path.read_bytes())


def _chunk_ranges(meta: ZarrMeta, region) -> list[range]:
    """For each axis, the chunk-grid indices touched by ``region`` (slices and integers)."""
    if len(region) > len(meta.shape):
        raise ValueError(f"region has {len(region)} axes, the array {len(meta.shape)}")
    ranges = []
    for axis, (size, chunk) in enumerate(zip(meta.shape, meta.chunks)):
        sel = region[axis] if axis < len(region) else slice(None)
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            if step < 0:
                raise ValueError("negative steps are not supported")
            ranges.append(range(start // chunk, (stop - 1) // chunk + 1) if stop > start else range(0))
        else:
            i = operator.index(sel)
            i += size if i < 0 else 0
            if not 0 <= i < size:
                raise IndexError(f"index {sel} is out of bounds for axis {axis} with size {size}")
            ranges.append(range(i // chunk, i // chunk + 1))
    return ranges


def mirror_zarr_region(remote: Remote, array_key: str, region: tuple[slice, ...]) -> Path:
    """Copy the metadata and every chunk file touching ``region`` into the cache; return the array's local path.

    A chunk the server does not have (404) is simply not created: zarr reads it as the fill value.
    """
    key = array_key.rstrip("/")
    meta = zarr_meta(remote, key)
    for meta_key in meta.meta_keys:
        remote.fetch(meta_key)                            # .zattrs may legitimately be absent
    for idx in itertools.product(*_chunk_ranges(meta, region)):
        remote.fetch(f"{key}/{meta.chunk_key(idx)}")
    return remote.local_path(key)


def read_zarr_region(remote: Remote, array_key: str, region: tuple[slice, ...]) -> np.ndarray:
    """The array's values over ``region`` (absent chunks read as the fill value)."""
    import zarr

    path = mirror_zarr_region(remote, array_key, region)
    return np.asarray(zarr.open_array(store=str(path), mode="r")[region])


def read_multiscales(remote: Remote, group_key: str) -> list[str]:
    """Paths of the pyramid levels of an OME-Zarr group, finest first, relative to the group.

    The order is the one of ``multiscales[0].datasets`` (OME-NGFF lists the highest resolution first).
    """
    key = group_key.rstrip("/")
    found = _fetch_first(remote, [f"{key}/.zattrs", f"{key}/zarr.json"])
    if found is None:
        raise FileNotFoundError(f"no zarr group at {key!r}")
    name, path = found
    doc = json.loads(path.read_bytes())
    if name.endswith(".zattrs"):
        multiscales = doc.get("multiscales")
    else:
        attributes = doc.get("attributes") or {}
        multiscales = (attributes.get("ome") or {}).get("multiscales") or attributes.get("multiscales")
    if not multiscales:
        raise ValueError(f"no multiscales metadata in {key}/{name.rpartition('/')[2]}")
    return [dataset["path"] for dataset in multiscales[0]["datasets"]]
