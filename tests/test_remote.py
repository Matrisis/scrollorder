import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from scrollorder.remote import (
    BUCKET_URL,
    CacheBudgetExceeded,
    Remote,
    default_cache_dir,
    http_get,
    mirror_zarr_region,
    parse_zarr_meta,
    read_multiscales,
    read_zarr_region,
    zarr_meta,
)
from tests.fakes import FakeBucket


def test_fetch_downloads_once_and_caches(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket"); bucket.put("A/x.bin", b"abc")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "cache", opener=bucket)
    p1, p2 = r.fetch("A/x.bin"), r.fetch("A/x.bin")
    assert p1 == p2 == tmp_path / "cache" / "A/x.bin"
    assert p1.read_bytes() == b"abc" and bucket.calls == 1

def test_fetch_404_returns_none_and_writes_nothing(tmp_path):
    r = Remote(base_url="https://h", cache_dir=tmp_path / "cache", opener=FakeBucket(tmp_path / "b"))
    assert r.fetch("missing") is None
    assert not (tmp_path / "cache" / "missing").exists()

def test_interrupted_download_leaves_no_file(tmp_path):
    def broken(url): raise http.client.IncompleteRead(b"ab")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "cache", opener=broken, retry_waits=(0, 0, 0))
    with pytest.raises(OSError):
        r.fetch("A/x.bin")
    assert list((tmp_path / "cache").rglob("*")) in ([], [tmp_path / "cache" / "A"])

def test_list_follows_continuation_tokens(tmp_path):
    bucket = FakeBucket(tmp_path / "b", page_size=2)
    for k in ("P/a/1", "P/a/2", "P/b/1", "P/c.json"): bucket.put(k, b"x")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    prefixes, keys = r.list("P/")
    assert prefixes == ["P/a/", "P/b/"] and [k for k, _ in keys] == ["P/c.json"]

def test_budget_is_enforced(tmp_path):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("a", b"x" * 10); bucket.put("b", b"x" * 10)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket, budget_bytes=15)
    r.fetch("a")
    with pytest.raises(CacheBudgetExceeded):
        r.fetch("b")

@pytest.mark.parametrize("zarr_format", [2, 3])
def test_missing_chunk_reads_as_fill_value(tmp_path, zarr_format):
    zarr = pytest.importorskip("zarr")
    data = np.arange(20 * 30 * 40, dtype=np.uint8).reshape(20, 30, 40)
    root = tmp_path / "b" / "S/vol.zarr/0"
    a = zarr.create_array(store=str(root), shape=data.shape, chunks=(8, 8, 8), dtype="uint8",
                          fill_value=0, zarr_format=zarr_format)
    a[:] = data
    bucket = FakeBucket(tmp_path / "b")
    bucket.delete_chunk("S/vol.zarr/0", (1, 1, 1))          # zone masquée : morceau absent
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    out = read_zarr_region(r, "S/vol.zarr/0", (slice(4, 18), slice(5, 25), slice(0, 40)))
    expected = data[4:18, 5:25, 0:40].copy()
    expected[8 - 4:16 - 4, 8 - 5:16 - 5, 8:16] = 0
    np.testing.assert_array_equal(out, expected)


# --- Remote: defaults, retries, budget, threads ----------------------------------------------

def _failing(errors, then=b"ok"):
    """Opener raising each error of ``errors`` in turn, then answering ``then``; records the URLs."""
    urls, pending = [], list(errors)
    def opener(url):
        urls.append(url)
        if pending:
            raise pending.pop(0)
        return then
    opener.urls = urls
    return opener

def test_default_cache_dir_is_resolved_when_the_remote_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("SCROLLORDER_CACHE", str(tmp_path / "env-cache"))
    assert default_cache_dir() == tmp_path / "env-cache"
    assert Remote().cache_dir == tmp_path / "env-cache"
    monkeypatch.delenv("SCROLLORDER_CACHE")
    assert Remote().cache_dir == Path.home() / ".cache" / "scrollorder"
    assert Remote().base_url == BUCKET_URL
    assert not (tmp_path / "env-cache").exists()           # creating a Remote writes nothing

def test_http_get_uses_urllib_with_a_plain_url(monkeypatch):
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return b"payload"
    def fake_urlopen(url, *args, **kwargs):
        seen.append(url)
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert http_get("https://h/k") == b"payload"
    assert seen == ["https://h/k"]            # a str, not a Request: no custom header, default User-Agent

def test_fetch_retries_with_the_configured_waits_then_gives_up(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr("scrollorder.remote.time.sleep", slept.append)
    opener = _failing([ConnectionResetError("x")] * 4)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener)
    with pytest.raises(ConnectionResetError):
        r.fetch("A/x.bin")
    assert len(opener.urls) == 4 and slept == [1.0, 4.0, 16.0]
    assert not (tmp_path / "c" / "A" / "x.bin").exists()

def test_fetch_retries_then_succeeds(tmp_path):
    opener = _failing([http.client.IncompleteRead(b"a"), TimeoutError("slow")], then=b"abc")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener, retry_waits=(0, 0, 0))
    p = r.fetch("A/x.bin")
    assert p.read_bytes() == b"abc" and len(opener.urls) == 3
    assert [q.name for q in (tmp_path / "c" / "A").iterdir()] == ["x.bin"]       # no .part left

def test_fetch_does_not_retry_a_404(tmp_path):
    err = urllib.error.HTTPError("https://h/k", 404, "Not Found", None, None)
    opener = _failing([err] * 5)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener)
    assert r.fetch("k") is None and len(opener.urls) == 1

def test_fetch_retries_other_http_errors_and_reports_them(tmp_path):
    err = urllib.error.HTTPError("https://h/k", 500, "Server Error", None, None)
    opener = _failing([err] * 5)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener, retry_waits=(0, 0))
    with pytest.raises(urllib.error.HTTPError) as info:
        r.fetch("k")
    assert info.value.code == 500 and len(opener.urls) == 3

@pytest.mark.parametrize("key", ["", "../x", "/x", "a/../b", "a//b", "a/"])
def test_fetch_refuses_keys_that_leave_the_cache(tmp_path, key):
    opener = _failing([])
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener)
    with pytest.raises(ValueError):
        r.fetch(key)
    assert opener.urls == []

def test_budget_counts_what_is_already_in_the_cache(tmp_path):
    cache = tmp_path / "c"
    (cache / "old").mkdir(parents=True)
    (cache / "old" / "f").write_bytes(b"x" * 10)
    bucket = FakeBucket(tmp_path / "b"); bucket.put("b", b"x" * 10)
    r = Remote(base_url="https://h", cache_dir=cache, opener=bucket, budget_bytes=15)
    assert r.cache_bytes() == 10
    with pytest.raises(CacheBudgetExceeded):
        r.fetch("b")
    assert not (cache / "b").exists() and not list(cache.rglob("*.part"))
    assert r.cache_bytes() == 10

def test_cache_bytes_follows_downloads(tmp_path):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("a/1", b"x" * 7); bucket.put("a/2", b"x" * 5)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    assert r.cache_bytes() == 0
    r.fetch("a/1"); r.fetch("a/2"); r.fetch("a/1")
    assert r.cache_bytes() == 12
    assert Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket).cache_bytes() == 12

def test_concurrent_fetches_of_one_key_are_counted_once(tmp_path):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("A/x.bin", b"x" * 10)
    barrier = threading.Barrier(4)
    def opener(url):
        barrier.wait(timeout=10)          # all four threads are downloading before any writes
        return bucket(url)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=opener, budget_bytes=15)
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(lambda _: r.fetch("A/x.bin"), range(4)))
    assert set(paths) == {tmp_path / "c" / "A" / "x.bin"}
    assert r.cache_bytes() == 10 and bucket.calls == 4
    assert [p.name for p in (tmp_path / "c" / "A").iterdir()] == ["x.bin"]


# --- Remote.list -----------------------------------------------------------------------------

def test_list_without_delimiter_returns_every_key_below_the_prefix(tmp_path):
    bucket = FakeBucket(tmp_path / "b", page_size=2)
    for k in ("P/a/1", "P/a/2", "P/b/1", "P/c.json", "Q/z"): bucket.put(k, b"xy")
    urls = []
    def spy(url): urls.append(url); return bucket(url)
    r = Remote(base_url="https://h/", cache_dir=tmp_path / "c", opener=spy)
    prefixes, keys = r.list("P/", delimiter=None)
    assert prefixes == []
    assert keys == [("P/a/1", 2), ("P/a/2", 2), ("P/b/1", 2), ("P/c.json", 2)]
    assert len(urls) == 2 and all("delimiter" not in u for u in urls)
    assert all(u.startswith("https://h/?list-type=2&prefix=P%2F") for u in urls)

def test_list_encodes_the_prefix_and_reports_sizes(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("P/a b+c/k&1", b"xyz"); bucket.put("P/a b+c/sub/1", b"x")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    prefixes, keys = r.list("P/a b+c/")
    assert prefixes == ["P/a b+c/sub/"] and keys == [("P/a b+c/k&1", 3)]

def test_list_of_an_unknown_prefix_is_empty(tmp_path):
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=FakeBucket(tmp_path / "b"))
    assert r.list("nothing/") == ([], [])

def test_list_at_the_root_of_the_bucket(tmp_path):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("S1/f", b"x"); bucket.put("top.txt", b"x")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    assert r.list("") == (["S1/"], [("top.txt", 1)])


# --- zarr metadata and chunk keys ------------------------------------------------------------

def _v2_json(**extra):
    return json.dumps({"zarr_format": 2, "shape": [20, 30, 40], "chunks": [8, 8, 8], "dtype": "|u1",
                       "fill_value": 0, "order": "C", "filters": None, "compressor": None, **extra}).encode()

def _v3_json(encoding=None, **extra):
    doc = {"zarr_format": 3, "node_type": "array", "shape": [20, 30, 40], "data_type": "uint8",
           "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8, 8]}},
           "fill_value": 0, "codecs": [{"name": "bytes"}], "attributes": {}, **extra}
    if encoding is not None:
        doc["chunk_key_encoding"] = encoding
    return json.dumps(doc).encode()

@pytest.mark.parametrize("meta, expected", [
    (dict(zarray=_v2_json()), "1.2.3"),
    (dict(zarray=_v2_json(dimension_separator=".")), "1.2.3"),
    (dict(zarray=_v2_json(dimension_separator="/")), "1/2/3"),
    (dict(zarr_json=_v3_json({"name": "default"})), "c/1/2/3"),
    (dict(zarr_json=_v3_json({"name": "default", "configuration": {"separator": "/"}})), "c/1/2/3"),
    (dict(zarr_json=_v3_json({"name": "default", "configuration": {"separator": "."}})), "c.1.2.3"),
    (dict(zarr_json=_v3_json({"name": "v2"})), "1.2.3"),
    (dict(zarr_json=_v3_json({"name": "v2", "configuration": {"separator": "/"}})), "1/2/3"),
])
def test_chunk_keys_follow_each_zarr_convention(meta, expected):
    assert parse_zarr_meta("A", **meta).chunk_key((1, 2, 3)) == expected

@pytest.mark.parametrize("raw, expected", [
    (None, 0.0), (0, 0.0), (7, 7.0), (0.5, 0.5), ("Infinity", np.inf), ("-Infinity", -np.inf),
])
def test_fill_values(raw, expected):
    meta = parse_zarr_meta("A", zarray=_v2_json(fill_value=raw))
    assert meta.fill_value == expected
    assert np.isnan(parse_zarr_meta("A", zarray=_v2_json(fill_value="NaN")).fill_value)

def test_parse_zarr_meta_describes_the_array():
    m = parse_zarr_meta("G/0", zarr_json=_v3_json({"name": "default"}, data_type="uint16", fill_value=3))
    assert (m.zarr_format, m.shape, m.chunks, m.dtype, m.fill_value) == (3, (20, 30, 40), (8, 8, 8), np.dtype("uint16"), 3.0)
    assert m.meta_keys == ("G/0/zarr.json",)
    m2 = parse_zarr_meta("G/0", zarray=_v2_json(dtype="<u2"))
    assert (m2.zarr_format, m2.dtype) == (2, np.dtype("<u2")) and m2.meta_keys == ("G/0/.zarray", "G/0/.zattrs")

def test_parse_zarr_meta_refuses_what_it_cannot_read():
    with pytest.raises(ValueError):
        parse_zarr_meta("A")
    with pytest.raises(ValueError):
        parse_zarr_meta("A", zarr_json=json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
    with pytest.raises(ValueError):
        parse_zarr_meta("A", zarr_json=_v3_json(chunk_grid={"name": "rectilinear", "configuration": {}}))

def test_zarr_meta_of_an_absent_array_raises(tmp_path):
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=FakeBucket(tmp_path / "b"))
    with pytest.raises(FileNotFoundError):
        zarr_meta(r, "nope/0")

@pytest.mark.parametrize("zarr_format", [2, 3])
def test_zarr_meta_reads_real_arrays_and_looks_locally_first(tmp_path, zarr_format):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "b" / "S/vol.zarr/0"
    zarr.create_array(store=str(root), shape=(20, 30, 40), chunks=(8, 8, 8), dtype="uint16",
                      fill_value=7, zarr_format=zarr_format)
    bucket = FakeBucket(tmp_path / "b")
    urls = []
    def spy(url): urls.append(url); return bucket(url)
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=spy)
    m = zarr_meta(r, "S/vol.zarr/0")
    assert (m.zarr_format, m.shape, m.chunks, m.dtype, m.fill_value) == (zarr_format, (20, 30, 40), (8, 8, 8), np.dtype("uint16"), 7.0)
    assert m.chunk_key((1, 0, 2)) == ("1.0.2" if zarr_format == 2 else "c/1/0/2")
    assert m.meta_keys == (("S/vol.zarr/0/.zarray", "S/vol.zarr/0/.zattrs") if zarr_format == 2 else ("S/vol.zarr/0/zarr.json",))
    n = len(urls)
    zarr_meta(r, "S/vol.zarr/0")
    assert len(urls) == n                  # second look: nothing asked of the server

def test_zarr_meta_of_a_sharded_array_is_the_shard_shape_and_a_missing_shard_reads_as_fill(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "b" / "S/sharded.zarr/0"
    a = zarr.create_array(store=str(root), shape=(20, 30, 40), chunks=(4, 4, 4), shards=(8, 8, 8),
                          dtype="uint8", fill_value=9, zarr_format=3)
    a[:] = 1
    bucket = FakeBucket(tmp_path / "b")
    bucket.delete_chunk("S/sharded.zarr/0", (0, 1, 2))
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    assert zarr_meta(r, "S/sharded.zarr/0").chunks == (8, 8, 8)
    out = read_zarr_region(r, "S/sharded.zarr/0", (slice(0, 8), slice(0, 16), slice(8, 24)))
    expected = np.ones((8, 16, 16), dtype=np.uint8)
    expected[:, 8:16, 8:16] = 9            # shard (0, 1, 2) covers [0:8, 8:16, 16:24]
    np.testing.assert_array_equal(out, expected)


# --- mirror_zarr_region ----------------------------------------------------------------------

def _chunk_indices(array_dir: Path):
    return {tuple(int(p) for p in f.relative_to(array_dir).parts[1:])
            for f in (array_dir / "c").rglob("*") if f.is_file()}

@pytest.mark.parametrize("region, wanted", [
    ((slice(0, 8), slice(0, 8), slice(0, 16)), {(0, 0, 0), (0, 0, 1)}),
    ((slice(-4, None), slice(0, 8), slice(0, 8)), {(2, 0, 0)}),
    ((5, slice(8, 9), slice(39, 40)), {(0, 1, 4)}),
    ((slice(4, 10), slice(None), slice(None)), {(z, y, x) for z in (0, 1) for y in range(4) for x in range(5)}),
    ((slice(4, 10),), {(z, y, x) for z in (0, 1) for y in range(4) for x in range(5)}),
    ((slice(5, 5), slice(0, 8), slice(0, 8)), set()),
])
def test_mirror_copies_metadata_and_only_the_chunks_that_touch_the_region(tmp_path, region, wanted):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "b" / "S/vol.zarr/0"
    a = zarr.create_array(store=str(root), shape=(20, 30, 40), chunks=(8, 8, 8), dtype="uint8",
                          fill_value=0, zarr_format=3)
    a[:] = 1
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=FakeBucket(tmp_path / "b"))
    path = mirror_zarr_region(r, "S/vol.zarr/0", region)
    assert path == tmp_path / "c" / "S/vol.zarr/0"
    assert (path / "zarr.json").is_file()
    assert _chunk_indices(path) == wanted

def test_mirror_of_a_v2_array_copies_zarray_and_zattrs(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "b" / "S/vol.zarr/0"
    a = zarr.create_array(store=str(root), shape=(20, 30, 40), chunks=(8, 8, 8), dtype="uint8",
                          fill_value=0, zarr_format=2)
    a[:] = 1
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=FakeBucket(tmp_path / "b"))
    path = mirror_zarr_region(r, "S/vol.zarr/0", (slice(0, 8), slice(0, 8), slice(0, 8)))
    assert (path / ".zarray").is_file() and (path / ".zattrs").is_file()
    assert sorted(p.name for p in path.iterdir() if p.name[0].isdigit()) == ["0.0.0"]

def test_mirror_tolerates_an_array_without_zattrs(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("A/.zarray", _v2_json(shape=[8, 8, 8]))
    bucket.put("A/0.0.0", b"")
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    path = mirror_zarr_region(r, "A/", (slice(0, 8), slice(0, 8), slice(0, 8)))
    assert path == tmp_path / "c" / "A"
    assert not (path / ".zattrs").exists() and (path / "0.0.0").is_file()

@pytest.mark.parametrize("zarr_format", [2, 3])
def test_a_missing_chunk_is_never_created_locally_and_reads_as_the_fill_value(tmp_path, zarr_format):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "b" / "S/vol.zarr/0"
    a = zarr.create_array(store=str(root), shape=(20, 30, 40), chunks=(8, 8, 8), dtype="uint8",
                          fill_value=5, zarr_format=zarr_format)
    a[:] = 1
    bucket = FakeBucket(tmp_path / "b")
    bucket.delete_chunk("S/vol.zarr/0", (1, 1, 1))
    assert bucket.calls == 0                           # preparing the fixture is not a file read
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    out = read_zarr_region(r, "S/vol.zarr/0", (slice(8, 16), slice(8, 16), slice(8, 16)))
    assert out.shape == (8, 8, 8) and (out == 5).all()
    meta = zarr_meta(r, "S/vol.zarr/0")
    assert not (tmp_path / "c" / "S/vol.zarr/0" / meta.chunk_key((1, 1, 1))).exists()


# --- read_multiscales ------------------------------------------------------------------------

def test_read_multiscales_of_a_v2_group(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("G.zarr/.zattrs", json.dumps({"multiscales": [
        {"version": "0.4", "datasets": [{"path": "0"}, {"path": "1"}, {"path": "2"}]}]}).encode())
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    assert read_multiscales(r, "G.zarr") == ["0", "1", "2"]

def test_read_multiscales_of_a_v3_group(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("G.zarr/zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{"datasets": [{"path": "s0"}, {"path": "s1"}]}]}}}).encode())
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    assert read_multiscales(r, "G.zarr/") == ["s0", "s1"]

def test_read_multiscales_errors(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("NoScales.zarr/zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode())
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    with pytest.raises(FileNotFoundError):
        read_multiscales(r, "Absent.zarr")
    with pytest.raises(ValueError):
        read_multiscales(r, "NoScales.zarr")


# --- the fake itself -------------------------------------------------------------------------

def test_fake_bucket_serves_counts_and_deletes(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("A/.zarray", _v2_json(shape=[8, 8], chunks=[8, 8]))
    bucket.put("A/0.0", b"chunk")
    assert bucket("https://h/A/0.0") == b"chunk" and bucket.calls == 1
    bucket.delete_chunk("A", (0, 0))
    assert bucket.calls == 1
    with pytest.raises(urllib.error.HTTPError) as info:
        bucket("https://h/A/0.0")
    assert info.value.code == 404 and bucket.calls == 1
    with pytest.raises(FileNotFoundError):
        bucket.delete_chunk("A", (0, 0))                # nothing left to delete: a wrong key fails loudly


# --- hardening (fix round 1) -----------------------------------------------------------------

def test_a_failed_write_leaves_no_partial_file_and_the_next_fetch_starts_over(tmp_path, monkeypatch):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("A/x.bin", b"abc"); bucket.put("B", b"12345")
    cache = tmp_path / "c"
    r = Remote(base_url="https://h", cache_dir=cache, opener=bucket)
    r.fetch("B")
    assert r.cache_bytes() == 5
    def disk_full(src, dst): raise OSError("disk full")
    with monkeypatch.context() as patch:
        patch.setattr("scrollorder.remote.os.replace", disk_full)
        with pytest.raises(OSError, match="disk full"):
            r.fetch("A/x.bin")
    assert [p for p in cache.rglob("*") if p.is_file()] == [cache / "B"]    # no .part, no target
    assert r.cache_bytes() == 5
    assert r.fetch("A/x.bin").read_bytes() == b"abc"                        # a new attempt starts from zero
    assert r.cache_bytes() == 8 and bucket.calls == 3

def test_each_writer_uses_its_own_partial_file_name(tmp_path, monkeypatch):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("A/x.bin", b"abc")
    cache = tmp_path / "c"
    sources, real_replace = [], os.replace
    def spy(src, dst):
        sources.append(Path(src))
        real_replace(src, dst)
    monkeypatch.setattr("scrollorder.remote.os.replace", spy)
    Remote(base_url="https://h", cache_dir=cache, opener=bucket).fetch("A/x.bin")
    (cache / "A" / "x.bin").unlink()
    Remote(base_url="https://h", cache_dir=cache, opener=bucket).fetch("A/x.bin")   # a second writer
    first, second = sources
    assert first != second
    for src in sources:
        assert src.parent == cache / "A" and src.name.startswith("x.bin.") and src.name.endswith(".part")
    assert [p.name for p in (cache / "A").iterdir()] == ["x.bin"]

def _answer(body):
    return Remote(base_url="https://h", cache_dir=Path("unused"), opener=lambda url: body)

def test_list_refuses_a_non_utf8_answer_carrying_a_dtd():
    xml = ('<?xml version="1.0" encoding="UTF-16"?>'
           '<!DOCTYPE ListBucketResult [<!ENTITY e "boom">]>'
           '<ListBucketResult><Name>&e;</Name><IsTruncated>false</IsTruncated></ListBucketResult>')
    with pytest.raises(OSError):
        _answer(xml.encode("utf-16")).list("P/")

def test_list_refuses_a_utf8_answer_carrying_a_dtd():
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<!DOCTYPE ListBucketResult [<!ENTITY e "boom">]>'
           '<ListBucketResult><Name>&e;</Name><IsTruncated>false</IsTruncated></ListBucketResult>')
    with pytest.raises(OSError):
        _answer(xml.encode("utf-8")).list("P/")

@pytest.mark.parametrize("contents", [
    "<Contents><Size>3</Size></Contents>",                         # no Key
    "<Contents><Key>k</Key><Size>three</Size></Contents>",         # Size is not a number
])
def test_list_raises_oserror_on_a_malformed_contents_entry(contents):
    xml = f"<ListBucketResult><IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
    with pytest.raises(OSError):
        _answer(xml.encode("utf-8")).list("P/")
