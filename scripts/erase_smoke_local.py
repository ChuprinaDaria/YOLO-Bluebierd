"""CPU sanity test for erase_vehicle_from_rgb.

Picks 5 frames from a tank_qwen output dir (close/mid/far + random),
runs cv2.inpaint, saves side-by-side (orig vs erased) для візуальної перевірки
що силует танку зник без halo/smear.

Usage:
    python scripts/erase_smoke_local.py <dataset_root> [--out data/runs/erase_sanity_<date>/]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont
from datasetforge.pipelines.shared.cond import erase_vehicle_from_rgb


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
    ap.add_argument("root", type=Path, help="Dataset root з images/ + vehicle_masks/ + metadata/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--method", default="TELEA", choices=["TELEA", "NS"])
    ap.add_argument("--inf-size", type=int, default=1024)
    args = ap.parse_args()

    out_dir = args.out or Path("data/runs") / f"erase_sanity_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT: {out_dir.resolve()}")

    # Pick frames: 2 close (<300m), 2 mid (300-600m), 1 far (>600m)
    by_zone = {"close": [], "mid": [], "far": []}
    for md_path in sorted((args.root / "metadata").glob("*.json")):
        d = json.loads(md_path.read_text()).get("distance_m")
        if d is None:
            continue
        if d < 300:
            by_zone["close"].append(md_path.stem)
        elif d < 600:
            by_zone["mid"].append(md_path.stem)
        else:
            by_zone["far"].append(md_path.stem)
    picks = by_zone["close"][:2] + by_zone["mid"][:2] + by_zone["far"][:1]
    print(f"picks: {picks}")

    inf_size = (args.inf_size, args.inf_size)
    for stem in picks:
        rgb_path = args.root / "images" / f"{stem}.jpg"
        mask_path = args.root / "vehicle_masks" / f"{stem}.png"
        md = json.loads((args.root / "metadata" / f"{stem}.json").read_text())

        rgb_pil = Image.open(rgb_path).convert("RGB").resize(inf_size, Image.LANCZOS)
        erased, stats = erase_vehicle_from_rgb(
            rgb_pil, mask_path, inf_size,
            dilate_px=args.dilate, radius=args.radius, method=args.method,
        )

        banner(rgb_pil := rgb_pil.copy(), f"ORIG {stem} d={md['distance_m']}m mask_px={stats['mask_px']}")
        banner(erased := erased.copy(), f"ERASED dilate={args.dilate} r={args.radius} {args.method}")
        canvas = Image.new("RGB", (rgb_pil.width * 2 + 4, rgb_pil.height), "white")
        canvas.paste(rgb_pil, (0, 0))
        canvas.paste(erased, (rgb_pil.width + 4, 0))
        canvas.save(out_dir / f"{stem}_erase.jpg", quality=92)
        print(f"  {stem}: mask_px={stats['mask_px']}  saved")

    print(f"\n  view: xdg-open {out_dir}")


if __name__ == "__main__":
    main()
