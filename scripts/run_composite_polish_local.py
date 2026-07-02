"""Local Stage 4 (composite) + Stage 5 (polish) runner — CPU-only.

Reads <root>/{images, ai_bg, vehicle_masks, normals, metadata}/ → writes
<root>/{composite, final}/. Pure numpy/cv2/Albumentations. No GPU required.

Used after pod runs Qwen Stage 3 only (split-compute).

Usage:
    python scripts/run_composite_polish_local.py <root> --cfg <yaml>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasetforge.pipelines.shared import composite, polish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cfg", type=Path, required=True,
                    help="YAML config (диффузійний — `diffusion.relight`, `polish` блоки)")
    ap.add_argument("--no-assert-identity", action="store_true",
                    help="Skip composite pixel-identity check (relight=ON логує drift, не throw)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    diff_cfg = cfg.get("diffusion", {})
    polish_cfg = cfg.get("polish", {})

    for sub in ("composite", "final"):
        (args.root / sub).mkdir(exist_ok=True)

    ai_bg_dir = args.root / "ai_bg"
    stems = sorted(p.stem for p in ai_bg_dir.glob("*.png"))
    print(f"composite+polish: {len(stems)} frames")
    print(f"  diffusion.relight.enabled = {(diff_cfg.get('relight') or {}).get('enabled', False)}")
    print(f"  polish.enabled = {polish_cfg.get('enabled', True)}")

    t0 = time.time()
    for i, stem in enumerate(stems):
        ts = time.time()
        composite.composite_one(
            rgb_path=args.root / "images" / f"{stem}.jpg",
            ai_bg_path=ai_bg_dir / f"{stem}.png",
            mask_path=args.root / "vehicle_masks" / f"{stem}.png",
            normals_path=args.root / "normals" / f"{stem}.png",
            meta_path=args.root / "metadata" / f"{stem}.json",
            out_path=args.root / "composite" / f"{stem}.png",
            diffusion_cfg=diff_cfg,
            assert_pixel_identity=not args.no_assert_identity,
        )
        md = json.loads((args.root / "metadata" / f"{stem}.json").read_text())
        seed = int(md.get("seed", 0)) + int(polish_cfg.get("seed_offset", 9000))
        polish.polish_one(
            args.root / "composite" / f"{stem}.png",
            args.root / "final" / f"{stem}.png",
            polish_cfg,
            seed=seed,
        )
        print(f"  [{i+1}/{len(stems)}] {stem}  {time.time()-ts:.1f}s")
    print(f"\n  ✓ done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
