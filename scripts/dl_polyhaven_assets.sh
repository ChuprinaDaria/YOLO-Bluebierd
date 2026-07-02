#!/usr/bin/env bash
# Завантажує Poly Haven HDRI + ground textures у локальний datasetforge/assets/.
# URL-и резолвиться через https://api.polyhaven.com (CC0).
#
# Якщо якийсь slug повертає 404 — зайди на https://polyhaven.com/hdris або /textures,
# знайди потрібний slug у URL сторінки (формат /a/<slug>) і заміни нижче.
#
# Usage: bash scripts/dl_polyhaven_assets.sh

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="datasetforge/assets"
mkdir -p "$ROOT/hdri"/{summer,autumn_mud,winter,spring}
mkdir -p "$ROOT/textures/ground"/{summer,autumn_mud,winter,spring}

# === HDRI slugs (1 per season). Якщо 404 — заміни у браузері. ===
declare -A HDRI=(
  [summer]="kloofendal_43d_clear_puresky"
  [autumn_mud]="kloppenheim_06_puresky"
  [winter]="passendorf_snow"
  [spring]="belfast_sunset_puresky"
)

# === Texture slugs (1 per season). ===
declare -A TEX=(
  [summer]="aerial_grass_rock"
  [autumn_mud]="brown_mud_leaves_01"
  [winter]="snow_03"
  [spring]="forrest_ground_01"
)

resolve() {
  # $1=slug $2=key path (json dot-path, e.g. "hdri.2k.hdr.url" or "Diffuse.2k.jpg.url")
  local slug="$1" path="$2"
  curl -s --max-time 15 "https://api.polyhaven.com/files/${slug}" |
    python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in '${path}'.split('.'):
    if not isinstance(d, dict) or k not in d:
        sys.exit(1)
    d = d[k]
print(d)
"
}

dl() {
  local url="$1" dst="$2"
  if [[ -f "$dst" && -s "$dst" ]]; then
    echo "[skip] $dst"; return
  fi
  echo "[get ] $url"
  if curl -L --fail --max-time 300 -o "$dst" "$url"; then
    echo "       -> $dst ($(stat -c%s "$dst" 2>/dev/null || echo ?) bytes)"
  else
    echo "[FAIL] $url"; rm -f "$dst"
  fi
}

# ---- HDRIs (2K .hdr) ----
for season in "${!HDRI[@]}"; do
  slug="${HDRI[$season]}"
  url=$(resolve "$slug" "hdri.2k.hdr.url" || true)
  if [[ -z "$url" ]]; then
    echo "[WARN] HDRI $slug not found via API (404 or renamed). Замість нього зайди https://polyhaven.com/hdris і знайди свій."
    continue
  fi
  dl "$url" "$ROOT/hdri/$season/${slug}_2k.hdr"
done

# ---- Ground textures (diffuse + normal, 2K JPG) ----
for season in "${!TEX[@]}"; do
  slug="${TEX[$season]}"
  base="$ROOT/textures/ground/$season"
  d_url=$(resolve "$slug" "Diffuse.2k.jpg.url" || true)
  n_url=$(resolve "$slug" "nor_gl.2k.jpg.url" || true)
  if [[ -z "$d_url" ]]; then
    echo "[WARN] texture $slug Diffuse not found via API. Поправ slug."
    continue
  fi
  dl "$d_url" "$base/${slug}_diff_2k.jpg"
  [[ -n "$n_url" ]] && dl "$n_url" "$base/${slug}_nor_gl_2k.jpg"
done

echo
echo "[done] assets у $ROOT/"
echo "       drag&drop $ROOT/ → Drive/MyDrive/yolo_bb/assets/"
