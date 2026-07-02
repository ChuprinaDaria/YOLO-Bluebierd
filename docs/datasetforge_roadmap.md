# datasetforge roadmap (Blender 3D composite)

> Замінив попередній diffusion-roadmap 2026-06-24. Деталі рішення — план `~/.claude/plans/abundant-snacking-thacker.md`.
> Причина flip: diffusion (FLUX/SDXL/GLIGEN+LoRA) не дає bbox labels — а вся цінність synthetic data для detection полягає у тому, що генератор вже знає де об'єкт. Composite render з 3D-меша → projection → bbox автоматом.

## Стек

```
Tool             Role
─────────────────────────────────────────────────────
Blender 4.2 LTS  render engine (Cycles, GPU)
BlenderProc      python wrapper з YOLO/COCO export
bpycv            instance segmentation + occlusion
Poly Haven HDRI  sky lighting CC0
Polycam scans    photogrammetry terrain
Sketchfab CC-BY  3D models техніки (free для 9/10 класів)
TurboSquid/CGT   radar_ew моделі (~$160 разово)
HF Jobs L4       batch render compute
Google Colab T4  sanity render compute (free)
```

## Phase 0 — Env audit + Colab notebook ✅ **PASS** (2026-06-24)

- Install Blender 4.x + BlenderProc у `datasetforge/pipelines/colab/blender_smoke_kaggle.ipynb`.
- Download GAZ Tigr models (Sketchfab CC-BY) + Poly Haven HDRI/ground у `datasetforge/assets/`.
- Прогнати BlenderProc Suzanne quickstart → GPU OPTIX/CUDA рендерить, HDF5 пишеться.

**G0 verify:** ✅ Suzanne (мавпа) рендериться, HDF5 пишеться, OPTIX GPU detected на Colab T4 і Kaggle P100.

## Phase 1 — `light_vehicle` smoke ✅ **PASS** (2026-06-25)

- Дописати `engine/scene_builder.py:build_scene` (BlenderProc loader + HDRI + camera).
- Дописати `engine/bbox_extractor.py:extract_from_3d_object` (BlenderProc COCO writer → YOLO TXT).
- Render 20 кадрів GAZ Tigr на 4 сезони × oblique+nadir.

**G1 verify:** ✅ 20/20 кадрів — техніка стоїть прямо, у кадрі, без втоплення в землю. Bbox tight.

**Робочий setup (reference для решти класів):**

| Параметр | Значення |
|---|---|
| Notebook | `datasetforge/pipelines/colab/blender_smoke_kaggle.ipynb` (Kaggle P100, kernel `dariachuprina/blender-smoke-light-vehicle` v5) |
| Config | `datasetforge/configs/v1_light_vehicle.yaml` |
| Class | `light_vehicle` id=7, реальна довжина ~4м, normalize до 5м max-dim |
| Models | 3 × GAZ Tigr `.glb` (Sketchfab) у `assets/models/light_vehicle/` |
| Image size | 640×640 |
| `distance_m` | `[150, 200, 300, 400, 500]` (3D line-of-sight камера→техніка) |
| `view_angle_deg` | `[10, 15, 20, 25, 30, 90]` (10-30° oblique + 90° nadir) |
| `hfov_deg` | 15° (ISR telephoto) |
| Seasons | summer / autumn_mud / winter / spring (HDRI + PBR ground з Poly Haven) |
| Lighting | HDRI strength=2.0 + explicit SUN energy=5.0 |
| Ground plane | 5000×5000 м (horizon clear на 800м drone) |
| Cycles samples | 64, GPU OPTIX/CUDA |
| Кадрів | 20 smoke |

**Критичний фікс (без нього 50%+ кадрів ламались):** після `bpy.ops.import_scene.gltf` ВИКЛИКАТИ `parent_clear(CLEAR_KEEP_TRANSFORM)` + `transform_apply(rotation, scale)`. glTF Y-up→Z-up корекція сидить на root EMPTY як π/2 X-rotation; без apply `set_rotation_euler` крутить vehicle навколо world Y → лягає на бік. Після фіксу local rotation = world rotation, yaw Z працює прямо.

**Як клонувати під наступний клас:** скопіювати yaml, поміняти `class.{name,id}`, скорегувати `distance_m` під tier (tank/AD 8-12м тримають 700/1000м, дрібні класи зрізають). Решта pipeline без змін.

## ~~Phase 1.5 — `ifv_apc` KS gate~~ SKIPPED

Відмовились 2026-06-24: `_synthetic_apc_726` від Дмитра вже є як готовий APC dataset, не дублюємо. KS-тест проти нього лишається у `datasetforge/tests/test_distribution_match.py` — використовується як sanity на Phase 3 кадрах.

## Phase 2 — Backplate compositor (Week 2-3, ~12 hr, ~$5)

Замінити PBR ground на real drone-photo backplate для більшої фотореалістичності.

- Розширити `scene_builder` параметром `backplate_path`.
- Blender compositor: 3D vehicle render з alpha → composite над real-photo backplate.
- Джерело backplates: inpaint існуючих кадрів з `data/external/sources_roboflow/` + `mendeley_uav/` через FLUX inpaint (прибрати техніку з кадру → отримати чистий drone-oblique фон).
- Sentinel-2 fallback для відсутніх сезонів.
- Прогнати existing `datasetforge/degradation/` post-render (JPEG quant, motion blur, atmosphere haze).

**G3 verify:** 20 наших vs 20 reference у шафл-сітці — reviewer accuracy < 70%.

## Phase 3 — Multi-class scale-out (Week 3-5, ~30 hr, ~$190)

- Підтягнути 8 решти класів (tank, artillery, air_defense, mlrs, truck_logistics, radar_ew, motorcycle, infantry).
- Один YAML config per class клонований з `v1_apc_reference.yaml`.
- Render 1000/class × 7 Tier-1 + 600/class × 3 Tier-2 ≈ 8800 positives.
- Перейти на HF Jobs L4.

**G4 verify:** per-class bbox histogram + count; each class ≥ 800 frames, median 50-80 px.

## Phase 4 — Hard negatives + assembly (Week 5-6, ~12 hr, $0)

- 2000 негативів: empty landscapes + procedural-destroyed Tier-1 (Boolean cuts + charred PBR) + civilian-truck-only frames.
- Final YOLO split via existing `output/`.

**G5 verify:** `ultralytics check_dataset` passes; YOLOv8n 50-епохах baseline mAP50 > 0.5 на 50-кадровому real-OSINT held-out smoke set.

## Compute & cost

| Item | Est. |
|---|---|
| Phase 0-2 Colab Free T4 | $0 |
| Phase 3 HF Jobs L4 batch ~12k frames | ~$20-30 |
| Phase 4 Hard negatives Colab | $0 |
| 3D models radar_ew (Krasukha + Kasta) | $160 |
| TurboSquid destroyed pack (опційно якщо procedural не задовольнить) | $99 |
| **Total min** | **~$180** |
| **Total з destroyed pack** | **~$280** |

## Timeline

**4-6 тижнів до v1.0.0** на одного programmistа.

## Що паралельно

- `training/` skeleton (yolo11 train loop, HF Jobs orchestration)
- `evaluation/` skeleton (mAP, per-class, real vs synthetic split)
- `inference/` + `aim_assist/` skeletons
- OSINT eval collection (~200-500 real drone frames)

## Що НЕ робимо

- ❌ Diffusion для object generation (зміст deprecated у `datasetforge/pipelines/_deprecated_diffusion/`)
- ❌ Купівля premium 3D pack ($2k+)
- ❌ Локальний рендер (Phenom II без AVX)
- ❌ Photogrammetry власна (Polycam-готових scans вистачить)

## Що НЕ блокує

- Brave1 timeline — паралельний трек
- Public datasets — використовуються як reference для матчингу 3D моделей + як джерело backgrounds (через inpaint)
