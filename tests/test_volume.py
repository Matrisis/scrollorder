import dataclasses

import numpy as np
import pytest

from scrollorder.dataconf import load_config
from scrollorder.remote import Remote
from scrollorder.volume import SliceImage, Volume, l0_to_level, level_to_l0, nearest_plane, px_mm
from tests.fakes import make_fake_sample, make_fake_volume, write_fake_config


def test_level_coordinates_follow_mean_pooling():
    assert l0_to_level(1.5, 2, "mean") == pytest.approx(0.0)
    assert l0_to_level(5.5, 2, "mean") == pytest.approx(1.0)
    assert level_to_l0(l0_to_level(123.4, 3, "mean"), 3, "mean") == pytest.approx(123.4)
    assert l0_to_level(8.0, 2, "stride") == pytest.approx(2.0)
    assert nearest_plane(5.4, 2, "mean") == 1

def test_dtype_max_follows_the_array(tmp_path):
    for dtype, top in (("uint8", 255.0), ("uint16", 65535.0)):
        vol = make_fake_volume(tmp_path / dtype, dtype=dtype)      # tests/fakes.py
        assert vol.dtype_max() == top

def test_read_slice_returns_the_plane_with_its_origin(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, (13, 64, 64), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data)
    sl = vol.read_slice(level=0, plane=5, bbox_xy_level=(10.0, 20.0, 30.0, 40.0), margin_px=3)
    assert sl.origin_xy == (7, 17)
    np.testing.assert_array_equal(sl.data, data[5, 17:44, 7:34].astype(np.float32))
    assert sl.z_l0 == pytest.approx(5.0)

def test_from_l0_averages_the_finest_level(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, (16, 64, 64), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data, pyramid="stride")
    sl = vol.read_slice(level=1, plane=3, bbox_xy_level=(4.0, 4.0, 10.0, 10.0), margin_px=0, from_l0=True)
    expected = data[6:8, 8:22, 8:22].reshape(2, 7, 2, 7, 2).mean(axis=(0, 2, 4))
    np.testing.assert_allclose(sl.data, expected)


def test_level_conversions_keep_types_and_reject_other_pyramids():
    assert isinstance(l0_to_level(5.5, 2, "mean"), float)
    assert isinstance(level_to_l0(1.0, 2, "stride"), float)
    z = np.array([1.5, 5.5, 9.5])
    out = l0_to_level(z, 2, "mean")
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])
    for pyramid in ("mean", "stride"):
        np.testing.assert_allclose(level_to_l0(l0_to_level(z, 3, pyramid), 3, pyramid), z)
    with pytest.raises(ValueError):
        l0_to_level(1.0, 1, "max")
    with pytest.raises(ValueError):
        level_to_l0(1.0, 1, "max")
    assert px_mm(0.002399, 3) == pytest.approx(0.019192)

def test_nearest_plane_rounds_half_up():
    assert nearest_plane(1.5, 1, "mean") == 1          # level coordinate 0.5 (Python's round gives 0)
    assert nearest_plane(2.0, 2, "stride") == 1        # 0.5 again
    assert nearest_plane(1.4, 1, "mean") == 0
    assert isinstance(nearest_plane(1.5, 1, "mean"), int)

def test_volume_refuses_an_unknown_pyramid_or_a_missing_level_0(tmp_path):
    vol = make_fake_volume(tmp_path)
    with pytest.raises(ValueError):
        dataclasses.replace(vol, pyramid="max")
    with pytest.raises(ValueError):
        dataclasses.replace(vol, levels={1: "1"})

@pytest.mark.parametrize("pyramid", ["mean", "stride"])
def test_level_1_shape_is_the_ceiling_of_half(tmp_path, pyramid):
    data = np.random.default_rng(1).integers(0, 255, (5, 7, 9), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data, pyramid=pyramid)
    assert vol.shape(0) == (5, 7, 9)
    assert vol.shape(1) == (3, 4, 5)
    level1 = vol.block(1, slice(None), slice(None), slice(None))
    if pyramid == "stride":
        expected = data[::2, ::2, ::2]
    else:
        expected = np.array([[[np.rint(data[2 * k:2 * k + 2, 2 * i:2 * i + 2, 2 * j:2 * j + 2].mean())
                               for j in range(5)] for i in range(4)] for k in range(3)])
    np.testing.assert_array_equal(level1, expected.astype(np.float32))

def test_block_is_float32_and_clipped_like_numpy(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, (13, 64, 64), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data)
    out = vol.block(0, slice(11, 20), slice(60, 90), slice(-100, 5))
    assert out.dtype == np.float32 and out.shape == (2, 4, 5)
    np.testing.assert_array_equal(out, data[11:20, 60:90, -100:5].astype(np.float32))
    assert vol.block(0, slice(70, 90), slice(0, 4), slice(0, 4)).shape == (0, 4, 4)

def test_read_slice_is_cropped_to_the_array(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, (13, 63, 61), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data)
    sl = vol.read_slice(0, 5, (-5.0, -5.0, 10.0, 10.0), 3)               # past the top-left corner
    assert sl.origin_xy == (0, 0)
    np.testing.assert_array_equal(sl.data, data[5, 0:14, 0:14].astype(np.float32))
    sl = vol.read_slice(0, 5, (55.0, 50.0, 70.0, 70.0), 2)               # past the bottom-right corner
    assert sl.origin_xy == (53, 48)
    np.testing.assert_array_equal(sl.data, data[5, 48:, 53:].astype(np.float32))
    sl = vol.read_slice(0, 5, (10.5, 20.2, 30.1, 40.9), 0)               # columns floor(x_min)..ceil(x_max)
    assert sl.origin_xy == (10, 20) and sl.data.shape == (22, 22)
    for outside in ((100.0, 100.0, 120.0, 120.0), (-50.0, -50.0, -20.0, -20.0)):
        with pytest.raises(ValueError):
            vol.read_slice(0, 5, outside, 2)
    for plane in (13, -1):
        for from_l0 in (False, True):
            with pytest.raises(IndexError):
                vol.read_slice(0, plane, (10.0, 10.0, 20.0, 20.0), 0, from_l0=from_l0)

def test_read_slice_at_a_coarser_level_reports_its_scale(tmp_path):
    vol = make_fake_volume(tmp_path)                                     # default data, "mean"
    sl = vol.read_slice(1, 3, (2.0, 4.0, 6.0, 8.0), 1)
    assert isinstance(sl, SliceImage) and sl.data.dtype == np.float32
    assert (sl.level, sl.plane, sl.origin_xy) == (1, 3, (1, 3))
    assert sl.z_l0 == pytest.approx(6.5)                                  # centre of L0 planes 6 and 7
    assert sl.px_mm == pytest.approx(vol.voxel_mm * 2)
    np.testing.assert_array_equal(sl.data, vol.block(1, slice(3, 4), slice(3, 10), slice(1, 8))[0])

def test_from_l0_averages_only_the_voxels_present_at_the_edges(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, (5, 7, 9), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data)                          # level 1 is (3, 4, 5)
    sl = vol.read_slice(1, 2, (0.0, 0.0, 100.0, 100.0), 0, from_l0=True)   # planes 4 and 5: only 4 exists
    assert sl.origin_xy == (0, 0) and sl.data.shape == (4, 5)
    expected = [[data[4:6, 2 * i:2 * i + 2, 2 * j:2 * j + 2].mean() for j in range(5)] for i in range(4)]
    np.testing.assert_allclose(sl.data, expected)
    assert sl.z_l0 == pytest.approx(4.5)

def test_from_l0_at_level_2_follows_the_level_grid(tmp_path):
    import zarr

    data = np.random.default_rng(0).integers(0, 255, (10, 13, 14), dtype=np.uint8)
    vol = make_fake_volume(tmp_path, data=data)
    zarr.create_array(store=str(vol.remote.opener.root / vol.group / "2"), shape=(3, 4, 4), chunks=(3, 4, 4),
                      dtype="uint8", fill_value=0, zarr_format=2)       # only its shape matters here
    vol2 = Volume(remote=vol.remote, group=vol.group, levels={0: "0", 2: "2"}, voxel_mm=vol.voxel_mm, pyramid="mean")
    sl = vol2.read_slice(2, 2, (0.0, 0.0, 10.0, 10.0), 0, from_l0=True)
    expected = [[data[8:12, 4 * i:4 * i + 4, 4 * j:4 * j + 4].mean() for j in range(4)] for i in range(4)]
    np.testing.assert_allclose(sl.data, expected)
    assert sl.z_l0 == pytest.approx(9.5)

def test_from_config_converts_the_level_keys(tmp_path):
    bucket = make_fake_sample(tmp_path / "bucket", "S", 1)               # zarr v3, one level
    cfg = load_config(write_fake_config(tmp_path, "S", 1))
    remote = Remote(base_url="https://h", cache_dir=tmp_path / "cache", opener=bucket)
    vol = Volume.from_config(cfg, remote)
    assert (vol.group, vol.levels, vol.voxel_mm, vol.pyramid) == ("S/volumes/v.zarr", {0: "0"}, 0.002399, "mean")
    assert vol.shape(0) == (13, 64, 64) and vol.dtype_max() == 255.0
    data = np.random.default_rng(0).integers(0, 255, (13, 64, 64), dtype=np.uint8)
    sl = vol.read_slice(0, 5, (10.0, 20.0, 30.0, 40.0), 0)
    np.testing.assert_array_equal(sl.data, data[5, 20:41, 10:31].astype(np.float32))
    two = dataclasses.replace(cfg, volume={**cfg.volume, "levels": {"0": "0", "1": "1"}})
    assert Volume.from_config(two, remote).levels == {0: "0", 1: "1"}
    wrong = dataclasses.replace(cfg, volume={**cfg.volume, "axes": "zyxc"})
    with pytest.raises(ValueError):
        Volume.from_config(wrong, remote)

def test_dtype_max_of_a_float_array_is_one(tmp_path):
    vol = make_fake_volume(tmp_path, dtype="float32")
    assert vol.dtype_max() == 1.0
    assert vol.block(0, slice(0, 2), slice(0, 2), slice(0, 2)).dtype == np.float32

def test_shape_and_dtype_are_read_once_per_level(tmp_path, monkeypatch):
    import scrollorder.volume as volume_module

    vol = make_fake_volume(tmp_path)
    keys, real = [], volume_module.zarr_meta
    monkeypatch.setattr(volume_module, "zarr_meta", lambda remote, key: keys.append(key) or real(remote, key))
    for _ in range(3):
        vol.shape(0), vol.shape(1), vol.dtype_max()
    assert keys == ["vol.zarr/0", "vol.zarr/1"]
