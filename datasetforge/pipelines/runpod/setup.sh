#!/bin/bash
# RunPod boot script для Blender+Flux smoke. Idempotent — re-running OK.
# Очікує:
#   - REPO_DIR (default /workspace/yolo-bluebierd)
#   - HF_TOKEN env var (через RunPod "Pod env vars" UI)
#   - Network volume mounted at /workspace/cache (для HF cache)

set -e

REPO_DIR="${REPO_DIR:-/workspace/yolo-bluebierd}"
CACHE_DIR="${HF_HOME:-/workspace/cache/huggingface}"

echo "[setup] REPO=$REPO_DIR HF_HOME=$CACHE_DIR"

# 1. HF cache → network volume (BEFORE будь-яких HF imports)
export HF_HOME="$CACHE_DIR"
mkdir -p "$HF_HOME"
if ! grep -qxF "export HF_HOME=$CACHE_DIR" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$CACHE_DIR" >> ~/.bashrc
fi

# 2. System deps (для cv2, matplotlib, git)
apt-get update -qq
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git wget

# 3. Python deps
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements_diffusion.txt"

# 4. HF whoami sanity — fail fast if token missing/invalid
python - <<'PY'
import os
from huggingface_hub import whoami
assert "HF_TOKEN" in os.environ, "HF_TOKEN не виставлений — set via RunPod 'Pod env vars'"
user = whoami(token=os.environ["HF_TOKEN"])["name"]
print(f"[hf] logged in as: {user}")
PY

# 5. Pre-warm Flux Depth-dev model на network volume.
# Один model (~24GB) — depth conditioning baked у ваги, окремий ControlNet не потрібен.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
cache = os.environ["HF_HOME"]
repo = "black-forest-labs/FLUX.1-Depth-dev"
print(f"[snapshot] {repo} (~24GB, перший раз 10-20 хв на 1Gbps лінк)")
# Без allow_patterns — fnmatch '*' НЕ matches '/', тому patterns скіпають
# weights у subdirectories (transformer/, text_encoder_2/ etc).
snapshot_download(repo, token=tok, cache_dir=cache)
print("[done] FLUX.1-Depth-dev cached")
PY

echo ""
echo "[setup] complete."
echo "→ Open notebook: $REPO_DIR/datasetforge/pipelines/runpod/notebook.ipynb"
