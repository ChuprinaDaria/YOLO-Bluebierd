# DEPRECATED 2026-06-24: diffusion cannot produce bbox labels.
# Kept for history. Engine pivoted to Blender 3D composite (BlenderProc).
# See datasetforge/README.md + plan abundant-snacking-thacker.md.
"""Прокатує prompts.jsonl через Gemini Imagen (gemini-2.5-flash-image, aka Nano Banana).

Без CLI/extension шарів — напряму через google-genai SDK з API key у .env.

Логи:
- success → image у images/{id}_v{n}.png + sidecar {id}_v{n}.json з метаданими
- блок цензурою → запис у censored.log з prompt_id, finish_reason, safety_ratings
- помилки API → у errors.log

Usage:
    python datasetforge/pipelines/gemini_sanity_runner.py \\
        data/sanity/v0_tank/prompts.jsonl \\
        --limit 2                  # для смок-тесту
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-image"


def load_prompts(jsonl_path: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Прокатати тільки перші N промптів (смок-тест).")
    ap.add_argument("--positives-only", action="store_true",
                    help="Пропускати hard_negative промпти.")
    ap.add_argument("--variations", type=int, default=None,
                    help="Override variations_requested з prompts.jsonl.")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[err] GEMINI_API_KEY не знайдено у .env", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)

    out_dir = args.prompts.parent
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    censored_log = out_dir / "censored.log"
    errors_log = out_dir / "errors.log"

    rows = load_prompts(args.prompts)
    if args.positives_only:
        rows = [r for r in rows if r["class_name"] != "hard_negative"]
    if args.limit:
        rows = rows[: args.limit]

    print(f"[run] model={MODEL}  prompts={len(rows)}")
    ok_count = 0
    censored_count = 0
    error_count = 0

    for i, row in enumerate(rows, 1):
        n_var = args.variations or row.get("variations_requested", 1)
        prompt = row["prompt"]
        pid = row["id"]
        print(f"[{i}/{len(rows)}] {pid}  ×{n_var}")

        for v in range(1, n_var + 1):
            stem = f"{pid}_v{v}"
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            except Exception as e:
                error_count += 1
                with errors_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": stem, "error": repr(e), "prompt": prompt},
                                       ensure_ascii=False) + "\n")
                print(f"  ✗ {stem}: API error → errors.log")
                continue

            candidate = (resp.candidates or [None])[0]
            if candidate is None:
                with censored_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": stem, "finish_reason": "NO_CANDIDATE",
                                        "prompt": prompt}, ensure_ascii=False) + "\n")
                censored_count += 1
                print(f"  ✗ {stem}: no candidate (likely blocked) → censored.log")
                continue

            image_bytes = None
            parts = (candidate.content.parts if candidate.content else []) or []
            for p in parts:
                if getattr(p, "inline_data", None) and p.inline_data.data:
                    image_bytes = p.inline_data.data
                    break

            if not image_bytes:
                finish_reason = str(getattr(candidate, "finish_reason", "UNKNOWN"))
                safety = []
                for sr in (getattr(candidate, "safety_ratings", None) or []):
                    safety.append({"category": str(sr.category), "prob": str(sr.probability)})
                with censored_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": stem, "finish_reason": finish_reason,
                                        "safety": safety, "prompt": prompt},
                                       ensure_ascii=False) + "\n")
                censored_count += 1
                print(f"  ✗ {stem}: no image, finish={finish_reason} → censored.log")
                continue

            img_path = img_dir / f"{stem}.png"
            img_path.write_bytes(image_bytes)
            sidecar = img_path.with_suffix(".json")
            sidecar.write_text(json.dumps({**row, "variation_idx": v,
                                           "model": MODEL,
                                           "image_path": str(img_path.name)},
                                          ensure_ascii=False, indent=2),
                               encoding="utf-8")
            ok_count += 1
            print(f"  ✓ {stem}: {len(image_bytes)} B")
            time.sleep(0.5)

    print()
    print(f"[done] ok={ok_count}  censored={censored_count}  errors={error_count}")
    if censored_log.exists():
        print(f"       censorship log → {censored_log}")
    if errors_log.exists():
        print(f"       error log      → {errors_log}")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
