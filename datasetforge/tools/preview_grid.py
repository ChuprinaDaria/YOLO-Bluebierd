"""Візуальна верифікація рендер-виходу ПЕРЕД дорогим RunPod-раном.

Будує контактку (grid) з усіх кадрів у out-папці:
  - RGB + bbox (aabb 5-col АБО obb 9-col — автодетект);
  - підпис із metadata: alt/angle/hfov, оцінка px цілі, visibility, оклюдери,
    прапорці WRECK / HARD-NEG;
  - опційно per-frame 4-up (RGB+bbox / vehicle_mask / depth / visibility-note).

Залежності: matplotlib, pillow, numpy (+ cv2 для depth-прев'ю). БЕЗ bpy/torch —
крутиться на будь-чому, у т.ч. локально після Blender-only smoke.

Приклади:
  python datasetforge/tools/preview_grid.py --out out/smoke --sheet out/smoke/_sheet.png
  python datasetforge/tools/preview_grid.py --out out/smoke --per-frame
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_meta(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _draw_labels(img: Image.Image, label_path: Path) -> tuple[Image.Image, str, int]:
    """Малює bbox (aabb або obb). Повертає (img, format, n_boxes)."""
    W, H = img.size
    d = ImageDraw.Draw(img)
    fmt, n = "empty", 0
    if not label_path.exists() or not label_path.read_text().strip():
        return img, fmt, n
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) == 5:  # aabb: cls xc yc w h
            fmt = "aabb"
            xc, yc, w, h = map(float, p[1:5])
            d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H,
                         (xc + w / 2) * W, (yc + h / 2) * H],
                        outline="lime", width=3)
            n += 1
        elif len(p) == 9:  # obb: cls x1 y1 x2 y2 x3 y3 x4 y4
            fmt = "obb"
            pts = [(float(p[1 + 2 * k]) * W, float(p[2 + 2 * k]) * H) for k in range(4)]
            d.line(pts + [pts[0]], fill="cyan", width=3)
            n += 1
    return img, fmt, n


def _caption(meta: dict, fmt: str, n: int) -> str:
    tags = []
    if meta.get("is_hard_negative"):
        tags.append("HARD-NEG")
    if meta.get("is_destroyed"):
        tags.append(f"WRECK:{meta.get('wreck_mode', '?')}")
    n_occ = meta.get("n_occluders", 0)
    vis = meta.get("visibility_fraction", 1.0)
    if n_occ:
        tags.append(f"occ={n_occ} vis={vis:.2f}")
    est = meta.get("est_target_px", [0, 0])
    head = (f"alt={meta.get('altitude_m', 0):.0f}m "
            f"ang={meta.get('view_angle_deg', 0):.0f}° "
            f"hfov={meta.get('hfov_deg', 0):.0f}° "
            f"d={meta.get('distance_m', 0):.0f}m")
    px = f"est~{est[0]:.0f}x{est[1]:.0f}px" if isinstance(est, (list, tuple)) else ""
    body = f"{fmt}:{n}  {px}"
    if tags:
        body += "  " + " ".join(tags)
    return head + "\n" + body


def build_sheet(out_dir: Path, sheet_path: Path, cols: int = 4,
                thumb: int = 420) -> None:
    img_dir = out_dir / "images"
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if not stems:
        print(f"[preview] no images in {img_dir}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(stems) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.9))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for k, stem in enumerate(stems):
        img = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
        img.thumbnail((thumb, thumb))
        img, fmt, n = _draw_labels(img, out_dir / "labels" / f"{stem}.txt")
        meta = _load_meta(out_dir / "metadata" / f"{stem}.json")
        axes[k].imshow(img)
        axes[k].set_title(_caption(meta, fmt, n), fontsize=8, family="monospace")

    plt.tight_layout()
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sheet_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[preview] contact sheet ({len(stems)} frames) → {sheet_path}")


def build_per_frame(out_dir: Path) -> None:
    """4-up на кадр: RGB+bbox / vehicle_mask / depth / caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cv2
    except Exception:
        cv2 = None

    img_dir = out_dir / "images"
    prev_dir = out_dir / "_preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    for stem in stems:
        img = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
        img, fmt, n = _draw_labels(img.copy(), out_dir / "labels" / f"{stem}.txt")
        meta = _load_meta(out_dir / "metadata" / f"{stem}.json")

        mask_p = out_dir / "vehicle_masks" / f"{stem}.png"
        depth_p = out_dir / "depth" / f"{stem}.png"
        mask = np.array(Image.open(mask_p).convert("L")) if mask_p.exists() else None
        depth = None
        if cv2 is not None and depth_p.exists():
            depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        ax[0].imshow(img); ax[0].set_title(f"RGB + {fmt}({n})"); ax[0].axis("off")
        if mask is not None:
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title(f"vehicle_mask visible_px={int((mask>=128).sum())}")
        ax[1].axis("off")
        if depth is not None:
            ax[2].imshow(depth, cmap="viridis")
            ax[2].set_title(f"depth max={int(depth.max())}")
        ax[2].axis("off")
        fig.suptitle(_caption(meta, fmt, n), family="monospace", fontsize=10)
        plt.tight_layout()
        fig.savefig(prev_dir / f"{stem}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"[preview] {len(stems)} per-frame 4-up → {prev_dir}")


def print_summary(out_dir: Path) -> None:
    """Текстова зведена таблиця — швидка перевірка розподілу без картинок."""
    metas = [_load_meta(p) for p in sorted((out_dir / "metadata").glob("*.json"))]
    if not metas:
        print(f"[preview] no metadata in {out_dir}")
        return
    n = len(metas)
    hn = sum(m.get("is_hard_negative", False) for m in metas)
    wreck = sum(m.get("is_destroyed", False) for m in metas)
    occ = sum(1 for m in metas if m.get("n_occluders", 0))
    empty = sum(1 for m in metas if m.get("n_boxes", 0) == 0)
    vises = [m.get("visibility_fraction", 1.0) for m in metas if m.get("n_occluders", 0)]
    mins = [m.get("est_target_px", [0, 0])[1] for m in metas if not m.get("is_hard_negative")]
    print(f"\n=== {out_dir} — {n} frames ===")
    print(f"  hard_neg      : {hn} ({100*hn/n:.0f}%)")
    print(f"  wreck         : {wreck} ({100*wreck/n:.0f}%)")
    print(f"  with occluders: {occ} ({100*occ/n:.0f}%)")
    print(f"  empty label   : {empty} (має дорівнювати hard_neg + wreck-hn)")
    if vises:
        print(f"  visibility    : min={min(vises):.2f} mean={sum(vises)/len(vises):.2f} "
              f"(усі мають бути ≥ min_visible_frac)")
    if mins:
        print(f"  est min-side  : min={min(mins):.1f}px max={max(mins):.1f}px "
              f"(усі ≥ min_side_px, інакше «крапки»)")
    fmts = {m.get("bbox_format", "?") for m in metas}
    print(f"  bbox_format   : {fmts}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="рендер-вихідна папка")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="куди зберегти контактку (default <out>/_sheet.png)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--per-frame", action="store_true",
                    help="ще й 4-up прев'ю на кожен кадр у <out>/_preview/")
    ap.add_argument("--no-sheet", action="store_true", help="лише текстова зведена")
    args = ap.parse_args(argv)

    print_summary(args.out)
    if not args.no_sheet:
        sheet = args.sheet or (args.out / "_sheet.png")
        build_sheet(args.out, sheet, cols=args.cols)
    if args.per_frame:
        build_per_frame(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
