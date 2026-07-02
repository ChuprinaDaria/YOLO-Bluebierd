"""Build a YOLO train/val/test dataset from a datasetforge run directory.

Reads:
    <run>/final/*.png     — polished frames (final output of stages 4-5)
    <run>/labels/*.txt    — YOLO labels (0-1 boxes per frame)
    <run>/metadata/*.json — per-frame metadata (distance_m, view_angle, season)

Writes to <out>/:
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt
    data.yaml

Split is **stratified by distance zone** so each split has a balanced mix of
near/mid/far frames. Zone bins:
    near   : distance_m ≤ 200
    mid    : 200 < distance_m ≤ 500
    far    : distance_m > 500
Ratios default 0.70 / 0.20 / 0.10.

Frames with empty labels are still copied (they act as easy hard-negatives —
YOLO handles empty label files correctly).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def zone_of(distance_m: float) -> str:
    if distance_m <= 200:
        return "near"
    if distance_m <= 500:
        return "mid"
    return "far"


def collect_frames(run: Path) -> list[dict]:
    frames = []
    for img_path in sorted((run / "final").glob("*.png")):
        stem = img_path.stem
        lbl = run / "labels" / f"{stem}.txt"
        meta = run / "metadata" / f"{stem}.json"
        if not lbl.exists() or not meta.exists():
            continue
        m = json.loads(meta.read_text())
        # n_boxes from actual label file (metadata sidecar is written at Stage 1
        # so it may be stale if relabel-from-masks ran afterwards).
        n_boxes = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        frames.append({
            "stem": stem,
            "img": img_path,
            "lbl": lbl,
            "distance_m": m["distance_m"],
            "zone": zone_of(m["distance_m"]),
            "n_boxes": n_boxes,
        })
    return frames


def stratified_split(frames: list[dict], ratios: tuple[float, float, float],
                     seed: int) -> dict[str, list[dict]]:
    """Split by zone so each zone honours the ratios independently."""
    rng = random.Random(seed)
    by_zone: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        by_zone[f["zone"]].append(f)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for zone, group in by_zone.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def write_split(splits: dict[str, list[dict]], out: Path) -> None:
    for split, frames in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for f in frames:
            shutil.copy2(f["img"], out / "images" / split / f["img"].name)
            shutil.copy2(f["lbl"], out / "labels" / split / f["lbl"].name)


def summarise(splits: dict[str, list[dict]]) -> None:
    print(f"{'split':<8} {'total':>6} {'near':>6} {'mid':>6} {'far':>6} {'w/box':>6} {'empty':>6}")
    for split, frames in splits.items():
        n = len(frames)
        n_near = sum(1 for f in frames if f["zone"] == "near")
        n_mid = sum(1 for f in frames if f["zone"] == "mid")
        n_far = sum(1 for f in frames if f["zone"] == "far")
        n_pos = sum(1 for f in frames if f["n_boxes"] > 0)
        n_emp = n - n_pos
        print(f"{split:<8} {n:>6} {n_near:>6} {n_mid:>6} {n_far:>6} {n_pos:>6} {n_emp:>6}")


def write_data_yaml(out: Path, class_names: list[str]) -> None:
    """data.yaml written with Kaggle-style relative paths.

    On Kaggle the dataset root will be mounted as `path`, so all image/label
    dirs resolve relative to it.
    """
    cfg = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="datasetforge run root (contains final/, labels/, metadata/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="YOLO dataset dir to create")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.20, 0.10],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-names", nargs="+", default=["tank"])
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {args.ratios}")

    frames = collect_frames(args.run)
    if not frames:
        raise SystemExit(f"no frames found under {args.run}")
    print(f"collected {len(frames)} frames from {args.run}")

    splits = stratified_split(frames, tuple(args.ratios), args.seed)
    if args.out.exists():
        raise SystemExit(f"output {args.out} exists — remove first")
    args.out.mkdir(parents=True)
    write_split(splits, args.out)
    write_data_yaml(args.out, args.class_names)
    summarise(splits)
    print(f"\n✓ dataset written to {args.out}")


if __name__ == "__main__":
    main()
