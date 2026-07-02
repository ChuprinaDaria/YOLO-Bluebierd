"""Side-by-side comparison: original 3D render vs Qwen-фон final.

Reads <root>/images/ (Blender) + <root>/final/ (Qwen polish) + <root>/labels/ (bbox).
Writes side-by-side 2-column PNG з bbox overlay у data/runs/<out>/.

Usage:
    python scripts/compare_renders.py <root> <out_dir> [--labels labels]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        n += 1
    return n


def banner(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 24], fill="black")
    draw.text((6, 4), text, fill="white", font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Dataset root з images/, final/, labels/")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--labels", default="labels")
    args = ap.parse_args()

    root = args.root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = root / "images"
    final_dir = root / "final"
    labels_dir = root / args.labels
    meta_dir = root / "metadata"
    assert images_dir.is_dir() and final_dir.is_dir(), \
        f"Missing images/ or final/ у {root}"

    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    print(f"comparing {len(stems)} frames")

    for stem in stems:
        blender = Image.open(images_dir / f"{stem}.jpg").convert("RGB")
        qwen_path = final_dir / f"{stem}.png"
        if not qwen_path.exists():
            qwen_path = final_dir / f"{stem}.jpg"
        if not qwen_path.exists():
            print(f"  skip {stem} — no final/")
            continue
        qwen = Image.open(qwen_path).convert("RGB")
        if qwen.size != blender.size:
            qwen = qwen.resize(blender.size, Image.LANCZOS)

        draw_bbox(blender, labels_dir / f"{stem}.txt")
        draw_bbox(qwen, labels_dir / f"{stem}.txt")

        md = json.loads((meta_dir / f"{stem}.json").read_text())
        banner(blender, f"BLENDER {stem} d={md['distance_m']}m {md['season']} {md['landscape']}")
        banner(qwen, f"QWEN-fon {stem} a={md['view_angle_deg']}d sun_el={md['sun_elevation_deg']:.0f}")

        canvas = Image.new("RGB", (blender.width * 2 + 4, blender.height), color="white")
        canvas.paste(blender, (0, 0))
        canvas.paste(qwen, (blender.width + 4, 0))
        canvas.save(args.out_dir / f"{stem}_cmp.jpg", quality=92)

    print(f"\nwrote to: {args.out_dir}")


if __name__ == "__main__":
    main()
