import importlib.util
import json
import os
import urllib.error
import urllib.parse

import pytest

from scrollorder import cli
from scrollorder.dataconf import DataConfig, load_config
from scrollorder.inventory import fetch_segments, inventory
from scrollorder.remote import Remote
from tests.fakes import FakeBucket, make_fake_sample, write_fake_config

needs_zarr = pytest.mark.skipif(importlib.util.find_spec("zarr") is None, reason="needs the optional zarr extra")


def _remote(tmp_path, bucket, **kw):
    return Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket, **kw)


class Recorder:
    """Wraps an opener and remembers the prefixes of the listings it was asked for."""

    def __init__(self, inner):
        self.inner, self.listed = inner, []

    def __call__(self, url):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        if "list-type" in query:
            self.listed.append(query["prefix"][0])
        return self.inner(url)


# --- inventory -------------------------------------------------------------------------------

@needs_zarr
def test_inventory_lists_volumes_segments_and_predictions(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)   # helper de tests/fakes.py
    r = Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket)
    inv = inventory(r, "S")
    assert [v["group"] for v in inv["volumes"]] == ["S/volumes/v.zarr"]
    assert inv["volumes"][0]["levels"][0]["shape"] == [13, 64, 64]
    assert [s["name"] for s in inv["segments"]] == ["s-w000_1", "s-w001_1"]
    assert "x.tif" in inv["segments"][0]["files"]
    assert inv["kollesis_keys"] == []


@needs_zarr
def test_inventory_never_lists_inside_zarr_folders(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1, with_surface_volume=True)
    inv = inventory(Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket), "S")
    assert all(".zarr/" not in f for f in inv["segments"][0]["files"])


@needs_zarr
def test_no_listing_request_goes_inside_a_zarr_folder(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2, with_surface_volume=True)
    bucket.put("S/stray.zarr/zarr.json", b"{}")                           # even straight under the sample
    spy = Recorder(bucket)
    inv = inventory(_remote(tmp_path, spy), "S")
    assert spy.listed and not any(".zarr/" in p for p in spy.listed)
    assert len(spy.listed) == len(set(spy.listed))                        # no folder is listed twice
    assert "S/segments/s-w000_1/surface-volumes/" in spy.listed          # it did walk down to the zarr's parent
    # the zarr folders are still reported, so that nothing disappears silently
    assert inv["segments"][0]["zarr_folders"] == ["surface-volumes/sv.zarr"]
    assert inv["segments"][0]["files"] == ["meta.json", "x.tif", "y.tif", "z.tif"]


@needs_zarr
def test_inventory_walks_the_real_nested_layout_and_reports_paths_relative_to_the_segment(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    seg = "S/segments/s-w000_1/"
    for folder in ("mesh/s-w000-on-a.tifxyz", "mesh/s-w000-on-b-t2.tifxyz"):
        for name in ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif"):
            bucket.put(f"{seg}{folder}/{name}", b"1")
    bucket.put(f"{seg}mesh/intermediate/notes.txt", b"1")
    bucket.put(f"{seg}ink-detection/pred.png", b"1")
    inv = inventory(_remote(tmp_path, bucket), "S")
    segment = inv["segments"][0]
    assert "mesh/s-w000-on-a.tifxyz/x.tif" in segment["files"]
    assert "mesh/intermediate/notes.txt" in segment["files"] and "ink-detection/pred.png" in segment["files"]
    assert segment["files"] == sorted(segment["files"])
    assert segment["tifxyz_folders"] == ["", "mesh/s-w000-on-a.tifxyz", "mesh/s-w000-on-b-t2.tifxyz"]
    assert segment["prefix"] == seg


@needs_zarr
def test_a_zarr_folder_without_the_zarr_suffix_is_recognised_by_its_metadata_and_not_descended_into(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    run = "S/segments/s-w000_1/ink-detection/run1/"
    bucket.put(run + "zarr.json", b"{}")
    for i in range(3):
        bucket.put(f"{run}c/{i}/0/0", b"1")
    spy = Recorder(bucket)
    segment = inventory(_remote(tmp_path, spy), "S")["segments"][0]
    assert not any(p.startswith(run + "c/") for p in spy.listed)
    assert segment["zarr_folders"] == ["ink-detection/run1"]
    assert not any(f.startswith("ink-detection/") for f in segment["files"])


@needs_zarr
def test_inventory_reads_every_listing_page(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=3)
    for i in range(7):
        bucket.put(f"S/segments/s-w001_1/extra/f{i}.bin", b"1")
    bucket.page_size = 2
    inv = inventory(_remote(tmp_path, bucket), "S")
    assert [s["name"] for s in inv["segments"]] == ["s-w000_1", "s-w001_1", "s-w002_1"]
    assert [f for f in inv["segments"][1]["files"] if f.startswith("extra/")] == [f"extra/f{i}.bin" for i in range(7)]


@needs_zarr
def test_inventory_describes_the_levels_and_the_group_attributes_of_a_volume(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    volume = inventory(_remote(tmp_path, bucket), "S")["volumes"][0]
    assert volume["name"] == "v.zarr"
    level = volume["levels"][0]
    assert level == {"path": "0", "key": "S/volumes/v.zarr/0", "zarr_format": 3, "shape": [13, 64, 64],
                     "chunks": [8, 32, 32], "dtype": "uint8", "fill_value": 0.0}
    assert volume["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] == "0"
    assert "error" not in volume
    json.dumps(volume, allow_nan=False)                                   # plain JSON, no numpy leftovers


@needs_zarr
def test_inventory_describes_a_prediction_array_with_its_attributes(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    predictions = inventory(_remote(tmp_path, bucket), "S")["predictions"]
    assert [p["group"] for p in predictions] == ["S/representations/predictions/lasagna/l.zarr"]
    pred = predictions[0]
    assert pred["levels"][0]["shape"] == [2, 13, 16, 16] and pred["levels"][0]["path"] == ""
    assert pred["levels"][0]["key"] == pred["group"]
    assert pred["attributes"] == {"channels": ["grad_mag", "cos"]}


@needs_zarr
def test_a_zarr_with_unreadable_metadata_is_reported_with_its_error_and_the_run_continues(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    bucket.put("S/volumes/odd.zarr/zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode())
    bucket.put("S/volumes/empty.zarr/chunk", b"1")
    inv = inventory(_remote(tmp_path, bucket), "S")
    by_name = {v["name"]: v for v in inv["volumes"]}
    assert sorted(by_name) == ["empty.zarr", "odd.zarr", "v.zarr"]
    assert by_name["odd.zarr"]["levels"] == [] and "multiscales" in by_name["odd.zarr"]["error"]
    assert by_name["empty.zarr"]["levels"] == [] and by_name["empty.zarr"]["error"]
    assert by_name["v.zarr"]["levels"][0]["shape"] == [13, 64, 64]


@needs_zarr
def test_kollesis_keys_are_the_listed_keys_and_folders_with_kolles_in_any_case(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    for key in ("S/transforms/Kollesis_joins.json", "S/segments/s-w000_1/mesh/KOLLES.txt",
                "S/representations/annotations/kollesis/a.json"):
        bucket.put(key, b"{}")
    bucket.put("S/representations/kollesis.zarr/zarr.json", b"{}")        # a zarr folder: its own key counts
    bucket.put("S/photos/p.jpg", b"1")
    inv = inventory(_remote(tmp_path, bucket), "S")
    assert inv["kollesis_keys"] == sorted([
        "S/transforms/Kollesis_joins.json", "S/segments/s-w000_1/mesh/KOLLES.txt",
        "S/representations/annotations/kollesis/", "S/representations/kollesis.zarr"])
    assert inv["top_level"] == ["photos", "representations", "segments", "transforms", "volumes"]


@needs_zarr
def test_folders_outside_the_three_walked_areas_are_listed_two_levels_deep_only(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    for depth in range(1, 7):
        bucket.put("S/transforms/" + "d/" * depth + "f.json", b"{}")
    spy = Recorder(bucket)
    inventory(_remote(tmp_path, spy), "S")
    assert [p for p in spy.listed if p.startswith("S/transforms/")] == ["S/transforms/", "S/transforms/d/"]


def _v2_array(bucket, folder, shape=(4, 4), fill_value=0, dtype="|u1", attrs=None):
    bucket.put(f"{folder}/.zarray", json.dumps({
        "zarr_format": 2, "shape": list(shape), "chunks": list(shape), "dtype": dtype, "fill_value": fill_value,
        "compressor": None, "filters": None, "order": "C"}).encode())
    if attrs is not None:
        bucket.put(f"{folder}/.zattrs", json.dumps(attrs).encode())


@needs_zarr
def test_a_prediction_without_the_zarr_suffix_is_found_by_its_metadata_and_never_opened(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    run = "S/representations/predictions/lasagna/20260102-L2"
    _v2_array(bucket, run, shape=(4, 4), attrs={"channels": ["grad_mag", "cos"]})
    bucket.put(f"{run}/0.0", b"1")
    bucket.put(f"{run}/sub/0.0", b"1")
    spy = Recorder(bucket)
    predictions = inventory(_remote(tmp_path, spy), "S")["predictions"]
    assert [p["group"] for p in predictions] == ["S/representations/predictions/lasagna/20260102-L2",
                                                 "S/representations/predictions/lasagna/l.zarr"]
    assert predictions[0]["levels"][0]["shape"] == [4, 4] and predictions[0]["levels"][0]["zarr_format"] == 2
    assert predictions[0]["attributes"] == {"channels": ["grad_mag", "cos"]}      # a format 2 array's .zattrs
    assert run + "/sub/" not in spy.listed


@needs_zarr
def test_a_prediction_one_level_deeper_inside_a_model_folder_is_found(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    bucket.put("S/representations/predictions/fibers/run3/a.zarr/zarr.json", json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [3, 5], "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3, 5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0, "codecs": [{"name": "bytes"}], "dimension_names": ["c", "x"],
        "attributes": {"channel_names": ["u", "v", "w"]}}).encode())
    predictions = {p["group"]: p for p in inventory(_remote(tmp_path, bucket), "S")["predictions"]}
    deep = predictions["S/representations/predictions/fibers/run3/a.zarr"]
    assert deep["levels"][0]["shape"] == [3, 5] and deep["levels"][0]["dimension_names"] == ["c", "x"]
    assert deep["attributes"] == {"channel_names": ["u", "v", "w"]}


def test_a_non_finite_fill_value_is_written_as_text_so_that_the_json_stays_valid(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("S/volumes/f.zarr/.zattrs", json.dumps({"multiscales": [{"datasets": [{"path": "0"}]}]}).encode())
    _v2_array(bucket, "S/volumes/f.zarr/0", shape=(2, 2), fill_value="NaN", dtype="<f4")
    volume = inventory(_remote(tmp_path, bucket), "S")["volumes"][0]
    assert volume["levels"][0]["fill_value"] == "NaN" and volume["levels"][0]["dtype"] == "float32"
    json.dumps(volume, allow_nan=False)


def test_a_broken_level_is_reported_in_place_and_the_other_levels_are_kept(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("S/volumes/g.zarr/.zattrs", json.dumps(
        {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}, {"path": "2"}]}]}).encode())
    _v2_array(bucket, "S/volumes/g.zarr/0", shape=(8, 8))
    _v2_array(bucket, "S/volumes/g.zarr/2", shape=(2, 2))
    volume = inventory(_remote(tmp_path, bucket), "S")["volumes"][0]
    assert [lv["path"] for lv in volume["levels"]] == ["0", "1", "2"]
    assert volume["levels"][0]["shape"] == [8, 8] and volume["levels"][2]["shape"] == [2, 2]
    assert "no zarr array" in volume["levels"][1]["error"] and "shape" not in volume["levels"][1]
    assert "error" not in volume


def test_a_refused_request_stops_the_inventory_instead_of_being_reported_as_bad_metadata(tmp_path):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("S/volumes/g.zarr/zarr.json", b"{}")

    def refuse_metadata(url):
        if url.endswith("zarr.json"):
            raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)
        return bucket(url)

    with pytest.raises(urllib.error.HTTPError):
        inventory(_remote(tmp_path, refuse_metadata, retry_waits=()), "S")


def test_inventory_of_an_unknown_sample_fails_instead_of_returning_an_empty_inventory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Nope"):
        inventory(_remote(tmp_path, FakeBucket(tmp_path / "b")), "Nope")


@needs_zarr
def test_inventory_is_plain_json(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2, with_surface_volume=True)
    inv = inventory(_remote(tmp_path, bucket), "S")
    assert json.loads(json.dumps(inv, allow_nan=False)) == inv
    assert set(inv) >= {"sample", "volumes", "segments", "predictions", "kollesis_keys"}


# --- load_config -----------------------------------------------------------------------------

def _config(**changes):
    cfg = {
        "sample": "S", "voxel_mm": 0.002399,
        "volume": {"group": "S/volumes/v.zarr", "levels": {"0": "0", "1": "1"}, "axes": "zyx"},
        "pyramid": "mean",
        "segments": [{"name": "w000", "prefix": "S/segments/a/"}],
        "tifxyz_to_l0": 1.0, "recto_offset_px": 0.0, "mass_from_l0": False, "overlay_ok": None,
    }
    for key, value in changes.items():
        if value is KeyError:
            del cfg[key]
        else:
            cfg[key] = value
    return cfg


def _write(tmp_path, cfg):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    return p


def test_load_config_rejects_a_missing_key(tmp_path):
    p = tmp_path / "c.json"; p.write_text(json.dumps({"sample": "S"}))
    with pytest.raises(ValueError, match="voxel_mm"):
        load_config(p)


def test_load_config_names_the_first_missing_key_in_schema_order(tmp_path):
    with pytest.raises(ValueError, match="^pyramid manquante$"):
        load_config(_write(tmp_path, _config(pyramid=KeyError, mass_from_l0=KeyError)))
    with pytest.raises(ValueError, match="^overlay_ok manquante$"):             # null is fine, absent is not
        load_config(_write(tmp_path, _config(overlay_ok=KeyError)))


def test_load_config_keeps_nested_objects_as_plain_dicts_and_defaults_optional_keys_to_none(tmp_path):
    cfg = load_config(_write(tmp_path, _config()))
    assert isinstance(cfg, DataConfig)
    assert (cfg.sample, cfg.voxel_mm, cfg.pyramid, cfg.tifxyz_to_l0) == ("S", 0.002399, "mean", 1.0)
    assert cfg.volume["group"] == "S/volumes/v.zarr" and cfg.volume["levels"] == {"0": "0", "1": "1"}
    assert cfg.segments[0]["prefix"] == "S/segments/a/"
    assert (cfg.recto_offset_px, cfg.mass_from_l0, cfg.overlay_ok) == (0.0, False, None)
    assert (cfg.umbilicus, cfg.lasagna, cfg.volume_9um, cfg.kollesis) == (None, None, None, None)


def test_load_config_reads_the_optional_keys(tmp_path):
    lasagna = {"level": 2, "grad_scale": 1000.0, "channels": {"grad_mag": {"key": "S/p.zarr", "index": 0}}}
    cfg = load_config(_write(tmp_path, _config(
        umbilicus="S/transforms/u.json", lasagna=lasagna, volume_9um={"pag0": "S/a/0", "pag50": "S/b/0"},
        kollesis="S/k.json", overlay_ok=True)))
    assert cfg.umbilicus == "S/transforms/u.json" and cfg.kollesis == "S/k.json" and cfg.overlay_ok is True
    assert cfg.lasagna["channels"]["grad_mag"]["index"] == 0 and cfg.volume_9um["pag50"] == "S/b/0"


@pytest.mark.parametrize("bad", ["avg", "", None, 2])
def test_load_config_rejects_an_unknown_pyramid_kind(tmp_path, bad):
    with pytest.raises(ValueError, match="pyramid"):
        load_config(_write(tmp_path, _config(pyramid=bad)))


def test_load_config_accepts_both_pyramid_kinds(tmp_path):
    assert load_config(_write(tmp_path, _config(pyramid="stride"))).pyramid == "stride"


def test_load_config_rejects_an_unknown_key_so_that_a_typo_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="lasgna"):
        load_config(_write(tmp_path, _config(lasgna={"level": 2})))


@pytest.mark.parametrize("changes, match", [
    ({"segments": []}, "segments"),
    ({"segments": [{"name": "w0"}]}, r"segments\[0\]\.prefix manquante"),
    ({"segments": [{"prefix": "S/a/"}]}, r"segments\[0\]\.name manquante"),
    ({"volume": {"levels": {"0": "0"}}}, r"volume\.group manquante"),
    ({"volume": {"group": "S/v.zarr"}}, r"volume\.levels manquante"),
])
def test_load_config_checks_the_nested_keys_later_tasks_index(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, _config(**changes)))


def test_load_config_reports_a_file_that_is_not_json(tmp_path):
    p = tmp_path / "c.json"; p.write_text("{not json")
    with pytest.raises(ValueError, match="c.json"):
        load_config(p)


# --- fetch_segments --------------------------------------------------------------------------

@needs_zarr
def test_fetch_segments_returns_folders_in_config_order(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    cfg = load_config(write_fake_config(tmp_path, sample="S", n_segments=2))
    dirs = fetch_segments(Remote(base_url="https://h", cache_dir=tmp_path / "c", opener=bucket), cfg)
    assert [d.name for d in dirs] == ["s-w000_1", "s-w001_1"]
    assert all((d / "x.tif").exists() and (d / "meta.json").exists() for d in dirs)


@needs_zarr
def test_fetch_segments_follows_the_order_of_the_config_not_the_server(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=3)
    path = write_fake_config(tmp_path, sample="S", n_segments=3)
    doc = json.loads(path.read_text()); doc["segments"].reverse(); path.write_text(json.dumps(doc))
    dirs = fetch_segments(_remote(tmp_path, bucket), load_config(path))
    assert [d.name for d in dirs] == ["s-w002_1", "s-w001_1", "s-w000_1"]
    assert dirs[0] == tmp_path / "c" / "S" / "segments" / "s-w002_1"


@needs_zarr
def test_fetch_segments_takes_mask_tif_when_it_exists_and_only_then(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    bucket.put("S/segments/s-w001_1/mask.tif", b"mask")
    dirs = fetch_segments(_remote(tmp_path, bucket), load_config(write_fake_config(tmp_path, "S", 2)))
    assert not (dirs[0] / "mask.tif").exists()
    assert (dirs[1] / "mask.tif").read_bytes() == b"mask"
    assert sorted(p.name for p in dirs[0].iterdir()) == ["meta.json", "x.tif", "y.tif", "z.tif"]


@needs_zarr
def test_fetch_segments_accepts_a_prefix_without_trailing_slash(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    path = write_fake_config(tmp_path, sample="S", n_segments=1)
    doc = json.loads(path.read_text()); doc["segments"][0]["prefix"] = "S/segments/s-w000_1"; path.write_text(json.dumps(doc))
    dirs = fetch_segments(_remote(tmp_path, bucket), load_config(path))
    assert dirs == [tmp_path / "c" / "S" / "segments" / "s-w000_1"] and (dirs[0] / "x.tif").is_file()


@needs_zarr
@pytest.mark.parametrize("missing", ["x.tif", "y.tif", "z.tif", "meta.json"])
def test_fetch_segments_names_the_segment_when_a_mandatory_file_is_missing(tmp_path, missing):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    (tmp_path / "b" / "S" / "segments" / "s-w001_1" / missing).unlink()
    cfg = load_config(write_fake_config(tmp_path, sample="S", n_segments=2))
    with pytest.raises(FileNotFoundError) as err:
        fetch_segments(_remote(tmp_path, bucket), cfg)
    assert "w001" in str(err.value) and missing in str(err.value) and "S/segments/s-w001_1/" in str(err.value)


@needs_zarr
def test_fetch_segments_reuses_the_cache(tmp_path):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    cfg = load_config(write_fake_config(tmp_path, "S", 2))
    r = _remote(tmp_path, bucket)
    fetch_segments(r, cfg)
    calls = bucket.calls
    fetch_segments(r, cfg)
    assert bucket.calls == calls


# --- command line ----------------------------------------------------------------------------

@needs_zarr
def test_inventory_command_writes_the_json_and_creates_the_folders(tmp_path, monkeypatch, capsys):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, bucket))
    out = tmp_path / "results" / "T0" / "S" / "inventory.json"
    cli.main(["inventory", "--sample", "S", "--out", str(out)])
    inv = json.loads(out.read_text())
    assert [s["name"] for s in inv["segments"]] == ["s-w000_1", "s-w001_1"]
    assert inv["volumes"][0]["levels"][0]["shape"] == [13, 64, 64]
    printed = capsys.readouterr().out
    assert str(out) in printed and "entries in error: 0" in printed


def test_inventory_command_explains_a_refused_listing(tmp_path, monkeypatch):
    def refuse(url):
        raise urllib.error.HTTPError(url, 403, "AccessDenied", None, None)

    monkeypatch.setattr(cli, "make_remote", lambda: Remote(base_url="https://h", cache_dir=tmp_path / "c",
                                                            opener=refuse, retry_waits=()))
    with pytest.raises(SystemExit) as err:
        cli.main(["inventory", "--sample", "S", "--out", str(tmp_path / "inv.json")])
    assert "403" in str(err.value) and not (tmp_path / "inv.json").exists()


@needs_zarr
def test_fetch_segments_command_prints_the_local_folders_in_order(tmp_path, monkeypatch, capsys):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=2)
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, bucket))
    cli.main(["fetch-segments", "--config", str(write_fake_config(tmp_path, "S", 2))])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [str(tmp_path / "c" / "S" / "segments" / f"s-w00{i}_1") for i in range(2)]


# --- fix round 1: non-finite numbers, output path checked first, entries in error shown ------------

def _no_json_constant(name):
    raise AssertionError(f"the file holds the JSON-invalid constant {name}")


def test_inventory_command_turns_non_finite_numbers_of_zarr_metadata_into_text(tmp_path, monkeypatch):
    bucket = FakeBucket(tmp_path / "b")
    # what real writers (numpy, python) emit: bare NaN / Infinity / -Infinity, also nested in lists
    bucket.put("S/volumes/n.zarr/.zattrs", (
        '{"multiscales": [{"datasets": [{"path": "0"}]}], '
        '"stats": {"min": NaN, "max": Infinity, "range": [-Infinity, 1.5], "big": 1e999}}').encode())
    _v2_array(bucket, "S/volumes/n.zarr/0", attrs=None)
    bucket.put("S/volumes/n.zarr/0/.zattrs", b'{"window": [NaN, {"hi": Infinity}]}')
    bucket.put("S/representations/predictions/p/a.zarr/zarr.json", json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [3, 5], "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3, 5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": float("nan"), "codecs": [{"name": "bytes"}],
        "attributes": {"stat": float("-inf"), "per_channel": [[float("nan")], [1.0]]}}).encode())
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, bucket))
    out = tmp_path / "inv.json"
    cli.main(["inventory", "--sample", "S", "--out", str(out)])
    inv = json.loads(out.read_text(), parse_constant=_no_json_constant)           # strict JSON
    volume = inv["volumes"][0]
    assert volume["attributes"]["stats"] == {"min": "NaN", "max": "Infinity", "range": ["-Infinity", 1.5],
                                             "big": "Infinity"}
    assert volume["levels"][0]["attributes"] == {"window": ["NaN", {"hi": "Infinity"}]}
    prediction = inv["predictions"][0]
    assert prediction["attributes"] == {"stat": "-Infinity", "per_channel": [["NaN"], [1.0]]}
    assert prediction["levels"][0]["fill_value"] == "NaN"


@pytest.mark.parametrize("case", ["parent_under_a_file", "output_is_a_folder"])
def test_inventory_command_fails_before_any_request_when_the_output_cannot_be_written(tmp_path, monkeypatch, case):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("S/x.json", b"{}")
    requests = []

    def opener(url):
        requests.append(url)
        return bucket(url)

    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, opener))
    if case == "parent_under_a_file":
        (tmp_path / "file").write_text("x")
        out = tmp_path / "file" / "sub" / "inv.json"
    else:
        out = tmp_path / "a_folder"
        out.mkdir()
    with pytest.raises(SystemExit) as err:
        cli.main(["inventory", "--sample", "S", "--out", str(out)])
    assert str(out) in str(err.value)
    assert requests == []                                   # the ~800 uncached requests were not started


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="permissions do not stop root")
def test_inventory_command_fails_before_any_request_when_the_output_folder_is_read_only(tmp_path, monkeypatch):
    bucket = FakeBucket(tmp_path / "b"); bucket.put("S/x.json", b"{}")
    requests = []
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, lambda url: requests.append(url) or bucket(url)))
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(SystemExit) as err:
            cli.main(["inventory", "--sample", "S", "--out", str(locked / "inv.json")])
    finally:
        locked.chmod(0o700)
    assert "locked" in str(err.value) and requests == []


@needs_zarr
def test_inventory_command_checks_the_output_folder_is_writable_then_leaves_no_probe_behind(tmp_path, monkeypatch):
    bucket = make_fake_sample(tmp_path / "b", sample="S", n_segments=1)
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, bucket))
    out = tmp_path / "new" / "deeper" / "inv.json"
    cli.main(["inventory", "--sample", "S", "--out", str(out)])
    assert [p.name for p in out.parent.iterdir()] == ["inv.json"]


def test_inventory_command_prints_the_entries_in_error_and_still_succeeds(tmp_path, monkeypatch, capsys):
    bucket = FakeBucket(tmp_path / "b")
    bucket.put("S/volumes/empty.zarr/chunk", b"1")                          # no metadata at all: the whole volume
    bucket.put("S/volumes/g.zarr/.zattrs", json.dumps(
        {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}).encode())
    _v2_array(bucket, "S/volumes/g.zarr/0", shape=(8, 8))                    # level 1 is missing: one level
    monkeypatch.setattr(cli, "make_remote", lambda: _remote(tmp_path, bucket))
    out = tmp_path / "inv.json"
    cli.main(["inventory", "--sample", "S", "--out", str(out)])               # no SystemExit: the exit code stays 0
    printed = capsys.readouterr().out
    assert "entries in error: 2" in printed
    assert "S/volumes/empty.zarr" in printed and "S/volumes/g.zarr/1" in printed
    assert "S/volumes/g.zarr/0" not in printed
    assert out.is_file()
