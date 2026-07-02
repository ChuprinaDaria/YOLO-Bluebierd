#!/usr/bin/env bash
# Зібрати assets bundle для Kaggle Dataset upload.
#
# Структура що збирається у /tmp/yolo-bb-df-assets/:
#   hdri/{summer,autumn_mud,winter,spring}/*.hdr|.exr
#   textures/ground/{summer,autumn_mud,winter,spring}/*_diff_*.png|.jpg
#   models/light_vehicle/*.glb
#   models/<other_class>/*.glb         (коли з'являться)
#
# Передумова: GAZ Tigr .glb лежать у Drive folder 1WdGkLf9H-FbscBWjkhwSHo1_L40VLqZU
# Скрипт сам не качає — щоб не залежати від `gdown`. Інструкція як докинути нижче.
#
# Usage:
#   bash scripts/prep_kaggle_assets.sh
#
# Результат:
#   /tmp/yolo-bb-df-assets/             — папка готова до Kaggle upload
#   /tmp/yolo-bb-df-assets.zip          — той самий бандл як zip (~50-200MB)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ASSETS="$REPO_ROOT/datasetforge/assets"
BUNDLE="/tmp/yolo-bb-df-assets"

echo "==> cleanup попереднього бандла"
rm -rf "$BUNDLE" "$BUNDLE.zip"

echo "==> copy HDRI + ground textures з $SRC_ASSETS"
mkdir -p "$BUNDLE"
cp -r "$SRC_ASSETS/hdri" "$BUNDLE/"
cp -r "$SRC_ASSETS/textures" "$BUNDLE/"
mkdir -p "$BUNDLE/models/light_vehicle"

echo
echo "==> models/light_vehicle/ — нагадування: glb файли НЕ автоматизую."
echo "    Файли GAZ Tigr / civilian / mil-pickup лежать у Drive folder:"
echo "    https://drive.google.com/drive/folders/1WdGkLf9H-FbscBWjkhwSHo1_L40VLqZU"
echo
echo "    Завантаж їх (через web Drive або gdown) у:"
echo "    $BUNDLE/models/light_vehicle/"
echo "    очікувані файли: gaz_tigr.glb, civilian_sedan.glb, mil_pickup.glb"
echo

echo "==> розмір бандла (без .glb моделей):"
du -sh "$BUNDLE"
echo
echo "==> після того як докинеш .glb моделі, спакуй командою:"
echo "    cd $(dirname $BUNDLE) && zip -r ${BUNDLE##*/}.zip ${BUNDLE##*/}"
echo
echo "==> upload bundle у Kaggle:"
echo "    1. kaggle.com → Datasets → New Dataset"
echo "    2. drag&drop $BUNDLE/ (вся папка) або $BUNDLE.zip"
echo "    3. Title:  yolo-bluebird-df-assets"
echo "    4. Slug:   yolo-bluebird-df-assets   (має точно збігатися — ipynb це використовує)"
echo "    5. Visibility: Private"
echo "    6. Create"
echo
echo "==> mount у Kaggle Notebook:"
echo "    Notebook → Add data → Your Datasets → yolo-bluebird-df-assets → Add"
echo "    Mount path: /kaggle/input/yolo-bluebird-df-assets/"
