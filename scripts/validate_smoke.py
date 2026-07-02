"""
Validate Blender-only smoke output (tank_smoke).

Checks: structure, label sanity, distance/angle histograms, metadata honesty,
bbox overlays on random frames, edge zooms across distance zones.

Usage:
    python scripts/validate_smoke.py <root_dir> [--labels labels|labels_min5|...]
"""
import argparse
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLOSE_BIN = (100, 250)
MID_BIN = (250, 600)
FAR_BIN = (600, 1100)


def ascii_hist(values, bin_edges, label, width=40):
    counts = [0] * (len(bin_edges) - 1)
    for v in values:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                break
    max_c = max(counts) if counts else 1
    print(f"\n  {label} (n={len(values)}):")
    for i, c in enumerate(counts):
        bar = "#" * int(c * width / max_c) if max_c else ""
        print(f"    [{bin_edges[i]:>5}-{bin_edges[i+1]:<5}) {c:>4d}  {bar}")


def draw_bbox(img, label_path, color="lime"):
    if not label_path.exists():
        return 0
    w, h = img.size
    draw = ImageDraw.Draw(img)
    n = 0
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, bw, bh = (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        n += 1
    return n


def annotate(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 22], fill="black")
    draw.text((4, 3), text, fill="white", font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--labels", default="labels", help="Labels subdir name (default: labels)")
    args = ap.parse_args()

    root = args.root
    assert root.is_dir(), f"{root} not found"
    labels_dir = root / args.labels
    assert labels_dir.is_dir(), f"{labels_dir} not found"

    suffix = "" if args.labels == "labels" else f"_{args.labels.replace('labels_', '')}"
    out_base = Path("data/runs") / f"validate_{root.name}_{date.today().isoformat()}{suffix}"
    out_overlay = out_base / "bbox_overlay"
    out_zoom = out_base / "edge_crops"
    out_overlay.mkdir(parents=True, exist_ok=True)
    out_zoom.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT: {out_base.resolve()}")
    print(f"LABELS: {labels_dir.name}/")

    # 1. STRUCTURE
    print("=" * 70)
    print("STRUCTURE")
    print("=" * 70)
    expected = ["images", "labels", "depth", "vehicle_masks", "normals", "metadata"]
    for sub in expected:
        p = root / sub
        n = len(list(p.glob("*"))) if p.exists() else 0
        ok = "OK" if n == 400 else "BAD"
        print(f"  {sub:<16} {n:>5}  [{ok}]")

    # 2. LABEL SANITY
    print("\n" + "=" * 70)
    print("LABELS")
    print("=" * 70)
    with_bbox = 0
    empty = 0
    multi = 0
    bbox_sizes_px = []
    for lbl in labels_dir.glob("*.txt"):
        lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
            continue
        if len(lines) == 1:
            with_bbox += 1
        else:
            multi += 1
        parts = lines[0].split()
        if len(parts) >= 5:
            bw_px = float(parts[3]) * 640
            bh_px = float(parts[4]) * 640
            bbox_sizes_px.append(min(bw_px, bh_px))

    total = with_bbox + empty + multi
    print(f"  total:      {total}")
    print(f"  with_bbox:  {with_bbox} ({100*with_bbox/total:.1f}%)")
    print(f"  empty:      {empty} ({100*empty/total:.1f}%)  <- dropped at min_side_px")
    print(f"  multi_bbox: {multi}")
    if bbox_sizes_px:
        bbox_sizes_px.sort()
        n = len(bbox_sizes_px)
        print(f"\n  bbox min_side_px:")
        print(f"    min:    {bbox_sizes_px[0]:.0f}")
        print(f"    p25:    {bbox_sizes_px[n//4]:.0f}")
        print(f"    median: {bbox_sizes_px[n//2]:.0f}")
        print(f"    p75:    {bbox_sizes_px[3*n//4]:.0f}")
        print(f"    max:    {bbox_sizes_px[-1]:.0f}")

    # 3. METADATA AGGREGATES
    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    distances, angles = [], []
    seasons, landscapes, models = Counter(), Counter(), Counter()
    diff_state = Counter()
    distance_to_nboxes = {}
    for md in (root / "metadata").glob("*.json"):
        data = json.loads(md.read_text())
        d = data.get("distance_m")
        a = data.get("view_angle_deg")
        if d is not None:
            distances.append(d)
            stem = md.stem
            lbl_file = labels_dir / f"{stem}.txt"
            has_bbox = lbl_file.exists() and bool(lbl_file.read_text().strip())
            distance_to_nboxes.setdefault(d, [0, 0])
            distance_to_nboxes[d][0 if has_bbox else 1] += 1
        if a is not None:
            angles.append(a)
        seasons[data.get("season")] += 1
        landscapes[data.get("landscape")] += 1
        models[data.get("model_variant")] += 1
        diff_state[data.get("diffusion", {}).get("enabled")] += 1

    ascii_hist(distances, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], "distance_m")

    print(f"\n  view_angle_deg counts: {dict(Counter(angles))}")
    print(f"  seasons:                {dict(seasons)}")
    print(f"  landscapes:             {dict(landscapes)}")
    print(f"  model_variants:         {dict(models)}")
    print(f"\n  diffusion.enabled:      {dict(diff_state)}  (expect: {{False: 400}} for Blender-only)")

    print(f"\n  bbox-coverage per distance:")
    print(f"    distance | with_bbox | empty")
    for d in sorted(distance_to_nboxes.keys()):
        wb, em = distance_to_nboxes[d]
        print(f"    {d:>6} m | {wb:>9} | {em:>5}")

    # 4. BBOX OVERLAYS — 20 random
    print("\n" + "=" * 70)
    print(f"BBOX OVERLAYS — {out_overlay}/")
    print("=" * 70)
    random.seed(42)
    all_imgs = sorted((root / "images").glob("*.jpg"))
    picks = random.sample(all_imgs, 20)
    for img_path in picks:
        stem = img_path.stem
        md = json.loads((root / "metadata" / f"{stem}.json").read_text())
        img = Image.open(img_path).convert("RGB")
        n = draw_bbox(img, labels_dir / f"{stem}.txt")
        tag = f"{stem} d={md['distance_m']}m a={md['view_angle_deg']}d boxes={n} {md['season']}"
        annotate(img, tag)
        img.save(out_overlay / f"{stem}_overlay.jpg", quality=90)
    print(f"  wrote 20 overlays")

    # 5. EDGE ZOOMS — close / mid / far × 5 each
    print("\n" + "=" * 70)
    print(f"EDGE ZOOMS — {out_zoom}/  (matting halo check)")
    print("=" * 70)

    bins = {"close": CLOSE_BIN, "mid": MID_BIN, "far": FAR_BIN}
    by_bin = {k: [] for k in bins}
    for md_path in sorted((root / "metadata").glob("*.json")):
        data = json.loads(md_path.read_text())
        if data.get("n_boxes", 0) == 0:
            continue
        d = data["distance_m"]
        for k, (lo, hi) in bins.items():
            if lo <= d < hi and len(by_bin[k]) < 5:
                by_bin[k].append(md_path.stem)

    for zone, stems in by_bin.items():
        for stem in stems:
            img_path = root / "images" / f"{stem}.jpg"
            lbl_path = labels_dir / f"{stem}.txt"
            lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            parts = lines[0].split()
            cx, cy, bw, bh = map(float, parts[1:5])
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            pad = 30
            x1 = int(max(0, (cx - bw / 2) * w - pad))
            y1 = int(max(0, (cy - bh / 2) * h - pad))
            x2 = int(min(w, (cx + bw / 2) * w + pad))
            y2 = int(min(h, (cy + bh / 2) * h + pad))
            crop = img.crop((x1, y1, x2, y2))
            scale = 3 if zone == "far" else 2
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
            md = json.loads((root / "metadata" / f"{stem}.json").read_text())
            annotate(crop, f"{zone} {stem} d={md['distance_m']}m {md['season']}")
            crop.save(out_zoom / f"{zone}_{stem}_zoom.jpg", quality=95)
        print(f"  {zone}: {len(stems)} zooms saved (x{3 if zone=='far' else 2})")

    print("\n" + "=" * 70)
    print("VIEW:")
    print(f"  xdg-open {out_overlay}/   # 20 random — bbox tightness")
    print(f"  xdg-open {out_zoom}/     # close/mid/far — matting halo")
    print("=" * 70)


if __name__ == "__main__":
    main()
