import csv
import json

import numpy as np
import pytest

from scrollorder.cli import main
from scrollorder.phantom import make_phantom
from tests.helpers import write_tifxyz

Z = 500.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A phantom written in the real formats: probability slice, TIFXYZ windings, umbilicus."""
    root = tmp_path_factory.mktemp("data")
    ph = make_phantom(seed=0, n_windings=14, n_theta=360)
    stack, (cx, cy) = ph["stack"], ph["center_xy"]
    np.save(root / "prob.npy", ph["image"].astype(np.float32))
    theta = 2 * np.pi * np.arange(360) / 360
    zs = np.linspace(Z - 10, Z + 10, 6)
    for k, radius in enumerate(stack):
        x = np.broadcast_to(cx + radius * np.cos(theta), (6, 360))
        y = np.broadcast_to(cy + radius * np.sin(theta), (6, 360))
        write_tifxyz(root / "windings" / f"w{k:03d}.tifxyz", x, y, np.broadcast_to(zs[:, None], (6, 360)))
    (root / "umbilicus.json").write_text(json.dumps({"control_points": [
        {"x": cx, "y": cy, "z": 0.0}, {"x": cx, "y": cy, "z": 1000.0}]}))
    return root, ph


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_stack_command_rebuilds_the_windings_from_tifxyz(dataset):
    root, ph = dataset
    out = root / "stack.npy"
    main(["stack", "--tifxyz", str(root / "windings"), "--z", str(Z), "--umbilicus", str(root / "umbilicus.json"),
          "--n-theta", "360", "--out", str(out), "--figure", str(root / "overlay.png"),
          "--prob", str(root / "prob.npy")])
    assert np.allclose(np.load(out), ph["stack"], atol=1e-3)
    assert (root / "overlay.png").stat().st_size > 10_000


def test_cut_command_writes_one_row_per_k_and_method(dataset):
    root, ph = dataset
    np.save(root / "truth.npy", ph["stack"])
    out = root / "cut.csv"
    main(["cut", "--prob", str(root / "prob.npy"), "--stack", str(root / "truth.npy"),
          "--umbilicus", str(root / "umbilicus.json"), "--z", str(Z), "--k", "2,4", "--theta-stride", "2",
          "--out", str(out)])
    rows = _read(out)
    assert [(r["K"], r["method"]) for r in rows] == [
        ("2", "interp"), ("2", "snap"), ("2", "cut"), ("4", "interp"), ("4", "snap"), ("4", "cut")]
    assert float(rows[-1]["accuracy"]) > 0.9


def test_cut_command_accepts_an_explicit_center_and_an_8_bit_tiff(dataset, tmp_path):
    import tifffile

    root, ph = dataset
    tifffile.imwrite(tmp_path / "prob.tif", (ph["image"] * 255).astype(np.uint8))
    np.save(tmp_path / "truth.npy", ph["stack"])
    cx, cy = ph["center_xy"]
    main(["cut", "--prob", str(tmp_path / "prob.tif"), "--stack", str(tmp_path / "truth.npy"),
          "--center", f"{cx},{cy}", "--k", "3", "--theta-stride", "4", "--out", str(tmp_path / "cut.csv")])
    assert float(_read(tmp_path / "cut.csv")[-1]["accuracy"]) > 0.9


def test_barcode_command_writes_one_row_per_band_and_method(dataset, tmp_path):
    root, ph = dataset
    np.save(tmp_path / "truth.npy", ph["stack"])
    main(["barcode", "--stack", str(tmp_path / "truth.npy"), "--bands-deg", "2,45", "--trials", "30",
          "--max-drop", "1", "--out", str(tmp_path / "barcode.csv")])
    rows = _read(tmp_path / "barcode.csv")
    assert [(float(r["band_deg"]), r["method"]) for r in rows] == [
        (2.0, "nearest"), (2.0, "barcode"), (45.0, "nearest"), (45.0, "barcode")]


def test_cut_command_requires_a_center(dataset, tmp_path):
    root, _ = dataset
    with pytest.raises(SystemExit):
        main(["cut", "--prob", str(root / "prob.npy"), "--stack", str(root / "prob.npy"), "--out", str(tmp_path / "x.csv")])
