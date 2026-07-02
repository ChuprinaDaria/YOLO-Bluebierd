# DEPRECATED 2026-06-24: diffusion cannot produce bbox labels.
# Kept for history. Engine pivoted to Blender 3D composite (BlenderProc).
# See datasetforge/README.md + plan abundant-snacking-thacker.md.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "diffusers>=0.30",
#   "transformers>=4.44",
#   "torch>=2.4",
#   "accelerate>=0.33",
#   "sentencepiece",
#   "protobuf",
#   "huggingface_hub>=0.25",
#   "Pillow",
# ]
# ///
"""CV-13.3 — FLUX.1-schnell sanity probe.

Runs on HF Jobs (l4x1 / a10g-small). Generates ONE drone-style frame
of T-72 tank in autumn rasputitsa landscape, pushes PNG + JSON sidecar
to private dataset Dariachup/yolo-bluebierd-data under
_sanity/v0_tank/flux_schnell/.

Acceptance (per CV-13.3): один кадр що візуально розпізнається.

Usage (from local):
    hf jobs uv run datasetforge/pipelines/hf_jobs/flux_sanity.py \\
        --flavor l4x1 --timeout 30m --secrets HF_TOKEN
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import FluxPipeline
from huggingface_hub import HfApi

assert "HF_TOKEN" in os.environ, "HF_TOKEN required (pass via --secrets HF_TOKEN)"

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
REPO_ID = "Dariachup/yolo-bluebierd-data"
REMOTE_DIR = "_sanity/v0_tank/flux_schnell"

PROMPT = (
    "oblique aerial drone reconnaissance photo, looking down at angle from above, "
    "from 400 meters altitude at 25 degrees angle above horizon. "
    "Single T-72B3 tank, stationary, on muddy dirt road. "
    "Scene: deep brown wet mud, bare leafless trees, overcast grey sky, "
    "rasputitsa, drizzle. "
    "Style: real combat footage style, low-quality compressed video frame, "
    "slight motion blur, slightly desaturated colors, EO daylight spectrum. "
    "Framing: centered single target, partially visible terrain context, "
    "no overlay text."
)
SEED = 42
STEPS = 4
GUIDANCE = 0.0  # FLUX-schnell
MAX_SEQ = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={device}")
if device == "cuda":
    print(f"[env] gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

print(f"[load] {MODEL_ID}")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
if device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)

print(f"[gen] steps={STEPS} guidance={GUIDANCE} seed={SEED}")
generator = torch.Generator("cpu").manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    guidance_scale=GUIDANCE,
    num_inference_steps=STEPS,
    max_sequence_length=MAX_SEQ,
    generator=generator,
).images[0]

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = Path("/tmp/flux_sanity")
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"tank_rasputitsa_400m_25deg_{ts}"
img_path = out_dir / f"{stem}.png"
image.save(img_path)
print(f"[save] {img_path} bytes={img_path.stat().st_size}")

meta = {
    "model": MODEL_ID,
    "prompt": PROMPT,
    "steps": STEPS,
    "guidance_scale": GUIDANCE,
    "seed": SEED,
    "max_sequence_length": MAX_SEQ,
    "timestamp_utc": ts,
    "image_size": list(image.size),
    "class_hint": "tank",
    "altitude_m_hint": 400,
    "view_angle_deg_hint": 25,
    "season_hint": "autumn_mud",
    "landscape_hint": "dirt_mud_road",
}
meta_path = img_path.with_suffix(".json")
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

api = HfApi(token=os.environ["HF_TOKEN"])
for p in (img_path, meta_path):
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=f"{REMOTE_DIR}/{p.name}",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"sanity: flux-schnell {stem}",
    )
    print(f"[push] -> {REPO_ID}/{REMOTE_DIR}/{p.name}")

print("[done]")
