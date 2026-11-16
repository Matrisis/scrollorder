"""Test-only fake of the public data bucket (anonymous HTTPS, S3 listing API).

No test may touch the network: everything that reads the server goes through
``FakeBucket`` handed to ``Remote(opener=...)``.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

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
