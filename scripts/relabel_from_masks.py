"""Re-extract YOLO labels from vehicle_masks/ with configurable min_side_px.

Writes у новий subdir (default labels_min5/), не чіпає original labels/.

Usage:
    python scripts/relabel_from_masks.py <dataset_root> [--min-side 5] [--out labels_min5]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_yolo_line(mask_path: Path, class_id: int, min_side_px: int):
    m = np.array(Image.open(mask_path))
    h_img, w_img = m.shape[:2]
    nz_y, nz_x = np.where(m > 0)
    if nz_y.size == 0:
        return None, 0
    x1, x2 = int(nz_x.min()), int(nz_x.max())
    y1, y2 = int(nz_y.min()), int(nz_y.max())
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    side = min(bw, bh)
    if side < min_side_px:
        return None, side
    xc = (x1 + bw / 2) / w_img
    yc = (y1 + bh / 2) / h_img
    return f"{class_id} {xc:.6f} {yc:.6f} {bw / w_img:.6f} {bh / h_img:.6f}\n", side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--min-side", type=int, default=5)
    ap.add_argument("--out", default=None, help="Output subdir name (default: labels_min<N>)")
    args = ap.parse_args()

    root = args.root
    out_name = args.out or f"labels_min{args.min_side}"
    out_dir = root / out_name
    out_dir.mkdir(exist_ok=True)

    print(f"min_side_px = {args.min_side}")
    print(f"out: {out_dir}")

    stats = Counter()
    by_distance = {}

    for mask_path in sorted((root / "vehicle_masks").glob("*.png")):
        stem = mask_path.stem
        md_path = root / "metadata" / f"{stem}.json"
        if not md_path.exists():
            continue
        md = json.loads(md_path.read_text())
        class_id = md["class_id"]
        d = md["distance_m"]
        line, side = mask_to_yolo_line(mask_path, class_id, args.min_side)
        out_path = out_dir / f"{stem}.txt"
        by_distance.setdefault(d, {"with_bbox": 0, "below_thr": 0, "no_pixels": 0})
        if line is None:
            out_path.write_text("")
            if side == 0:
                stats["no_pixels"] += 1
                by_distance[d]["no_pixels"] += 1
            else:
                stats["below_threshold"] += 1
                by_distance[d]["below_thr"] += 1
        else:
            out_path.write_text(line)
            stats["with_bbox"] += 1
            by_distance[d]["with_bbox"] += 1

    total = sum(stats.values())
    print(f"\ntotal: {total}")
    for k, v in stats.items():
        print(f"  {k}: {v} ({100 * v / total:.1f}%)")

    print(f"\nbbox-coverage per distance (min_side={args.min_side}):")
    print(f"  distance | with_bbox | below_thr | no_pixels")
    for d in sorted(by_distance):
        s = by_distance[d]
        print(f"  {d:>6} m | {s['with_bbox']:>9} | {s['below_thr']:>9} | {s['no_pixels']:>9}")


if __name__ == "__main__":
    main()
