#!/usr/bin/env bash
# Зібрати assets bundle для Kaggle Dataset upload — усі класи, які є локально.
#
# Структура що збирається у /tmp/yolo-bb-df-assets/:
#   hdri/{summer,autumn_mud,winter,spring}/*.hdr|.exr
#   textures/ground/{summer,autumn_mud,winter,spring}/*_diff_*.png|.jpg
#   models/<class>/*.glb        (усі папки з $REPO/datasetforge/assets/models/)
#
# Sesja 8+: tank pipeline. Скрипт бере ВСІ підпапки models/ — не хардкодить клас.
#
# Usage:
#   bash scripts/prep_kaggle_assets.sh
#
# Результат:
#   /tmp/yolo-bb-df-assets/             — папка готова до Kaggle upload
#   /tmp/yolo-bb-df-assets.zip          — той самий бандл як zip

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ASSETS="$REPO_ROOT/datasetforge/assets"
BUNDLE="/tmp/yolo-bb-df-assets"

echo "==> cleanup попереднього бандла"
rm -rf "$BUNDLE" "$BUNDLE.zip"

echo "==> copy HDRI + ground textures з $SRC_ASSETS"
mkdir -p "$BUNDLE"
[ -d "$SRC_ASSETS/hdri" ] && cp -r "$SRC_ASSETS/hdri" "$BUNDLE/"
[ -d "$SRC_ASSETS/textures" ] && cp -r "$SRC_ASSETS/textures" "$BUNDLE/"

echo "==> copy models/ (all classes)"
if [ -d "$SRC_ASSETS/models" ]; then
  cp -r "$SRC_ASSETS/models" "$BUNDLE/"
  echo "    included classes:"
  for d in "$BUNDLE/models"/*/; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name '*.glb' | wc -l)
    echo "     - $(basename "$d") ($n .glb)"
  done
else
  echo "    [warn] $SRC_ASSETS/models відсутній"
fi

echo "==> розмір бандла:"
du -sh "$BUNDLE"

echo
echo "==> dataset-metadata.json (для 'kaggle datasets version')"
cat > "$BUNDLE/dataset-metadata.json" <<META
{
  "title": "yolo-bluebird-df-assets",
  "id": "dariachuprina/yolo-bluebird-df-assets",
  "licenses": [{"name": "unknown"}]
}
META
cat "$BUNDLE/dataset-metadata.json"

echo
echo "==> upload:"
echo "    cd $BUNDLE && kaggle datasets version -p . -m 'sesja 9 — tank models + PR#3 assets' --dir-mode zip"
