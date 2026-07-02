# DEPRECATED 2026-06-24: diffusion cannot produce bbox labels.
# Kept for history. Engine pivoted to Blender 3D composite (BlenderProc).
# See datasetforge/README.md + plan abundant-snacking-thacker.md.
"""Будує матрицю промптів для Gemini Imagen sanity-probe на 1 клас.

Вхід:  YAML конфіг (див. datasetforge/configs/v0_<class>_sanity.yaml)
Вихід: JSONL з one-prompt-per-line + людський TXT для copy-paste у Gemini UI.

Usage:
    python datasetforge/pipelines/build_sanity_prompts.py datasetforge/configs/v0_tank_sanity.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def build_prompt(cls: str, model: str, state: str, corner: dict, season_key: str,
                 season_desc: str, style: dict) -> str:
    state_phrase = state.replace("_", " ")
    landscape = corner["landscape"].replace("_", " ")
    return (
        f"{style['view']}, from {corner['altitude_m']} meters altitude "
        f"at {corner['view_angle_deg']} degrees angle above horizon. "
        f"Single {model} {cls}, {state_phrase}, on {landscape} terrain. "
        f"Scene: {season_desc}. "
        f"Style: {style['quality']}, {style['modality']}. "
        f"Framing: {style['framing']}."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cls = cfg["class"]["name"]
    models = cfg["class"]["models"]
    states = cfg["class"]["state_options"]
    corners = cfg["camera_corners"]
    seasons = cfg["seasons"]
    style = cfg["style_anchors"]
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / cfg["output"]["out_file"]
    txt_path = jsonl_path.with_suffix(".txt")

    rows: list[dict] = []
    # Один model+state на (corner, season) — щоб не роздути матрицю. Беремо round-robin.
    pairs = [(m, s) for m in models for s in states]
    i = 0
    for corner in corners:
        for season_key, season_desc in seasons.items():
            model, state = pairs[i % len(pairs)]
            i += 1
            prompt = build_prompt(cls, model, state, corner, season_key, season_desc, style)
            rows.append({
                "id": f"{cls}_{corner['label']}_{season_key}",
                "class_id": cfg["class"]["id"],
                "class_name": cls,
                "model": model,
                "state": state,
                "altitude_m": corner["altitude_m"],
                "view_angle_deg": corner["view_angle_deg"],
                "landscape": corner["landscape"],
                "season": season_key,
                "expected_bbox_px": corner["expected_bbox_px"],
                "variations_requested": cfg["output"]["variations_per_prompt"],
                "prompt": prompt,
            })

    # Hard-negative probe (без техніки, ті ж сезони).
    for landscape in [c["landscape"] for c in corners]:
        for season_key, season_desc in seasons.items():
            for template in cfg["hard_negatives_probe"]:
                prompt = (
                    f"{style['view']}, oblique drone reconnaissance frame. "
                    f"{template.replace('Same landscape', landscape.replace('_', ' ') + ' terrain')}. "
                    f"Scene: {season_desc}. "
                    f"Style: {style['quality']}, {style['modality']}."
                )
                rows.append({
                    "id": f"hardneg_{landscape}_{season_key}_{template[:20].replace(' ', '_')}",
                    "class_id": None,
                    "class_name": "hard_negative",
                    "model": None,
                    "state": None,
                    "altitude_m": None,
                    "view_angle_deg": None,
                    "landscape": landscape,
                    "season": season_key,
                    "expected_bbox_px": None,
                    "variations_requested": 2,
                    "prompt": prompt,
                })

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with txt_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"### {r['id']}\n{r['prompt']}\n\n")

    print(f"[ok] {len(rows)} prompts → {jsonl_path}")
    print(f"[ok] human-readable copy → {txt_path}")
    positives = sum(1 for r in rows if r["class_name"] == cls)
    negatives = sum(1 for r in rows if r["class_name"] == "hard_negative")
    print(f"     positive: {positives}   hard_negative: {negatives}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
