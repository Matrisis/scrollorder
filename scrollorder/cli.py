"""Command line: python -m scrollorder {stack,cut,barcode} ...  (see README)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .experiments import evaluate_barcode, evaluate_ordered_cut
from .io_real import build_stack, umbilicus_center
from .polar import polar_resample


def _load_image(path) -> np.ndarray:
    """2D probability slice from .npy or .tif, scaled to [0, 1]."""
    path = Path(path)
    if path.suffix == ".npy":
        image = np.load(path)
    else:
        import tifffile

        image = tifffile.imread(path)
    image = np.asarray(image)
    if image.ndim != 2:
        raise SystemExit(f"{path}: expected a 2D slice, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(float) / np.iinfo(image.dtype).max
    image = image.astype(float)
    return image / image.max() if image.max() > 1.5 else image


def _center(args, parser) -> tuple[float, float]:
    if args.center:
        cx, cy = (float(v) for v in args.center.split(","))
        return cx, cy
    if args.umbilicus and args.z is not None:
        return umbilicus_center(args.umbilicus, args.z, args.umbilicus_scale)
    parser.error("give --center x,y or both --umbilicus and --z")


def _write_csv(rows: list[dict], path) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    print(f"-> {path}")


def _tifxyz_dirs(paths) -> list[Path]:
    dirs = []
    for p in map(Path, paths):
        dirs += [p] if (p / "x.tif").exists() else sorted(d for d in p.iterdir() if (d / "x.tif").exists())
    return dirs


def _overlay(image, stack, center, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = 2 * np.pi * np.arange(stack.shape[1]) / stack.shape[1]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for k, radius in enumerate(stack):
        ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
                ".", ms=1, color="tab:orange" if k % 2 else "tab:cyan")
    ax.plot(*center, "r+", ms=12)
    ax.set_title("verified windings over the probability slice: they must sit on the ridges")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _add_center_args(p) -> None:
    p.add_argument("--center", help="scroll axis in the slice, as x,y (pixels of the slice)")
    p.add_argument("--umbilicus", help="umbilicus JSON (control_points); needs --z")
    p.add_argument("--z", type=float, help="height of the slice, in the slice's own voxel units")
    p.add_argument("--umbilicus-scale", type=float, default=1.0,
                   help="factor from umbilicus coordinates to slice coordinates (default 1)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="scrollorder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stack", help="build the ground-truth stack of windings at one height from TIFXYZ surfaces")
    p.add_argument("--tifxyz", nargs="+", required=True, help="TIFXYZ folders, or folders containing them")
    p.add_argument("--xyz-scale", type=float, default=1.0,
                   help="factor from TIFXYZ coordinates to slice coordinates (default 1)")
    p.add_argument("--n-theta", type=int, default=1440)
    p.add_argument("--out", required=True)
    p.add_argument("--prob", help="probability slice, to draw the check figure")
    p.add_argument("--figure", help="PNG overlay of the windings on the slice (needs --prob)")
    _add_center_args(p)

    p = sub.add_parser("cut", help="experiment 1: hidden sheets between two known windings")
    p.add_argument("--prob", required=True, help="2D surface-probability slice (.npy or .tif)")
    p.add_argument("--stack", required=True, help="stack .npy from the `stack` command")
    p.add_argument("--k", default="2,5,10,20", help="numbers of turns between the anchors")
    p.add_argument("--dr", type=float, default=0.5, help="radial step of the polar image, in pixels")
    p.add_argument("--theta-stride", type=int, default=1)
    p.add_argument("--anchor-stride", type=int)
    p.add_argument("--pos-per-gap", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.35, help="minimum sheet spacing, as a share of the mean")
    p.add_argument("--prior-weight", type=float, default=0.05)
    p.add_argument("--out", required=True)
    _add_center_args(p)

    p = sub.add_parser("barcode", help="experiment 2: integer shift across a hidden angular band")
    p.add_argument("--stack", required=True)
    p.add_argument("--bands-deg", default="2,5,15,45,90")
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--max-drop", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "stack":
        center = _center(args, parser)
        if args.z is None:
            parser.error("--z is required to slice the surfaces")
        stack = build_stack(_tifxyz_dirs(args.tifxyz), args.z, center, args.n_theta, args.xyz_scale)
        if stack.size == 0:
            raise SystemExit("no surface crosses this height: check --z and --xyz-scale")
        np.save(args.out, stack)
        crossings = int((np.diff(stack, axis=0) <= 0).sum())
        print(f"{stack.shape[0]} windings, {100 * np.isfinite(stack).mean():.0f}% of rays covered, "
              f"{crossings} order violations (should be 0) -> {args.out}")
        if args.figure:
            if not args.prob:
                parser.error("--figure needs --prob")
            _overlay(_load_image(args.prob), stack, center, args.figure)
            print(f"check figure -> {args.figure}")

    elif args.command == "cut":
        center = _center(args, parser)
        stack = np.load(args.stack)
        polar = polar_resample(_load_image(args.prob), center, stack.shape[1], np.nanmax(stack) + 5, args.dr)
        rows = evaluate_ordered_cut(
            polar, args.dr, stack, [int(k) for k in args.k.split(",")],
            theta_stride=args.theta_stride, anchor_stride=args.anchor_stride,
            pos_per_gap=args.pos_per_gap, alpha=args.alpha, prior_weight=args.prior_weight)
        _write_csv(rows, args.out)

    elif args.command == "barcode":
        stack = np.load(args.stack)
        bands = [max(1, round(float(d) / 360 * stack.shape[1])) for d in args.bands_deg.split(",")]
        _write_csv(evaluate_barcode(stack, bands, n_trials=args.trials, max_drop=args.max_drop, seed=args.seed), args.out)


if __name__ == "__main__":
    main()
