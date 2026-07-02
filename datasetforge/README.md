# datasetforge

Синтетичний генератор кадрів через **Blender 3D composite render**: 3D-модель техніки → реальний дроновий фон/HDRI → render → автоматичний YOLO bbox з проєкції 3D-меша.

> Замінив попередній diffusion-pipeline (FLUX/SDXL/GLIGEN+LoRA) 2026-06-24. Причина: text2image не дає bbox labels — вся цінність синтетики для detection полягає у тому, що генератор *знає* де об'єкт. Diffusion давав ще одну купу нерозмічених картинок. Composite дає bbox автоматом з проєкції 3D-меша. Деталі в плані `~/.claude/plans/abundant-snacking-thacker.md`.

## Стек

| Шар | Інструмент | Чому |
|---|---|---|
| **Render engine** | [BlenderProc](https://github.com/DLR-RM/BlenderProc) (DLR) | Native COCO/YOLO bbox export, готовий Colab template, ~30 хв до першого кадру |
| **3D platform** | Blender 4.2 LTS | Стандарт, безкоштовний, потужний bpy API |
| **Instance masks** | [bpycv](https://github.com/DIYer22/bpycv) | Occlusion grading + depth для atmosphere matching |
| **3D models** | Sketchfab (CC-BY), BlenderKit Free, Polycam, GrabCAD | Безкоштовно для 9/10 класів; ~$160 на radar_ew |
| **HDRI/textures** | [Poly Haven](https://polyhaven.com) (CC0), Polycam | CC0 sky + ground PBR, per-season |
| **Compute** | Google Colab Free T4 (sanity) / HF Jobs L4 (batch) | Local Phenom II без AVX → Blender 4.x не запуститься локально |

## Чому це працює (для detection)

1. **3D-модель у Blender має точні мешеві координати.** Камера дивиться на сцену → проєкція 8 вершин 3D bbox → 2D pixel bbox. Безкоштовно, без розмітки.
2. **Composite а не повна 3D-сцена:** фон = реальне дронове фото / HDRI / Polycam terrain scan. Тільки техніка рендериться у 3D. Дешево, реалістично, контрольовано.
3. **Match style anchor `_synthetic_apc_726`:** камера-висота/кут/distortion, ground textures per season, distribution розмірів bbox (медіана 60 px) — все керується конфігом.

## Pipeline

```
configs/v1_<class>.yaml (camera × season × landscape × degradation)
      │
      ▼
┌──────────────────────────────────────────┐
│ engine/render_runner.py (Colab/HF Jobs) │
│                                          │
│  1. bproc.init()                         │
│  2. load .blend (assets/models/<cls>/)   │
│  3. ground plane + season texture        │
│  4. HDRI sky lighting (per season)       │
│  5. camera_sampler — altitude × angle    │
│  6. bproc.renderer.render() (Cycles)     │
│  7. bproc.writer.write_coco_annotations  │
│  8. bbox_extractor → YOLO TXT            │
│  9. FrameMetadata sidecar JSON           │
│  10. degradation/ post (JPEG/blur/atmo) │
└──────────┬───────────────────────────────┘
           ▼
      image + YOLO label + metadata.json
           ▼
   data/datasetforge_v1/{train,valid,test}/
```

## Цільові характеристики

| | Цільове |
|---|---|
| Image size | 1920×1080 — нативна роздільність сенсора (тренування: SAHI-тайли 640 або imgsz≥1280, БЕЗ downscale до 640 — див. docs/pixel_budget.md) |
| Камера | висота 150-200 м (робоча) / ~300 м (крейсер), HFOV 92° або 112° |
| Bbox min-side px | ≥6 (детекція, межа для крейсера @92°), ≥20 (класифікація) |
| Стратифікація | висота × кут × hfov × сезон × ландшафт |
| Метадані | JSON sidecar обовʼязковий |

## Папки

| | Призначення |
|---|---|
| `engine/` | BlenderProc wrappers (scene_builder, bbox_extractor, camera_sampler, season_lighting, render_runner) |
| `configs/` | Per-class YAML (`v1_apc_reference.yaml` = canonical) |
| `assets/models/<class>/` | `.blend` файли (gitignored, push у HF private dataset) |
| `assets/hdri/<season>/` | Poly Haven CC0 |
| `assets/textures/ground/<season>/` | Per-season ground PBR |
| `assets/backgrounds/` | Real drone backplates (inpainted from Roboflow) |
| `degradation/` | Post-render JPEG/blur/atmosphere |
| `output/` | YoloBox + FrameMetadata writers |
| `pipelines/colab/` | Notebook для smoke + batch на Colab Free T4 |
| `pipelines/hf_jobs/` | UV scripts + Dockerfile для batch на HF Jobs L4 |
| `pipelines/_deprecated_diffusion/` | Колишній FLUX/Gemini sanity-код, кепт для історії |
| `tests/` | KS-test distribution_match + bbox geometry + camera sampler |

## Roadmap (real)

| Phase | Зусилля | Що | Verify |
|---|---|---|---|
| 0 | 1 тиж | Colab BlenderProc setup + 1 модель + 1 HDRI | G0: `basic.py` не пустий PNG |
| 1 | 1-2 тиж | `light_vehicle` smoke (GAZ Tigr CC-BY) — 20 кадрів | G1: bbox ±3 px to silhouette |
| 1.5 | 3 дні | `ifv_apc` KS gate (BTR-80) — 200 кадрів | G2: KS p>0.05 vs `_synthetic_apc_726` |
| 2 | 2-3 тиж | Стратифікація + degradation parity | G3: ourside-vs-anchor шафл guess < 70% |
| 3 | 3-5 тиж | Решта 8 класів batch на HF Jobs L4 | G4: per-class ≥800 кадрів, median 50-80px |
| 4 | 5-6 тиж | Hard negatives + assembly + train smoke | G5: YOLOv8n mAP50 > 0.5 on real OSINT |

**Total ~4-6 тижнів. Бюджет ~$200** (radar_ew моделі $160 + HF Jobs L4 ~$30 batch).

## Compute estimate

- Colab Free T4: 0$. 12 hr/тиждень. ОК для Phase 0-2 sanity (<2k кадрів).
- HF Jobs L4: ~$30 на повний batch 12k кадрів. Phase 3-4.
- Local: **немає** (Phenom II без AVX).

## Що НЕ робимо

- ❌ Diffusion для object generation (FLUX/SDXL/GLIGEN/LoRA) — не дає labels. Деприкейчено у `pipelines/_deprecated_diffusion/`.
- ❌ Купівля premium 3D pack ($2k+) — overkill для синтетики. Free-first + $160 точкова покупка radar_ew.
- ❌ Photogrammetry власна — Polycam-готових scans вистачить як baseline.
- ❌ Локальний рендер — Phenom II без AVX, Blender 4.x не запуститься.

## Стан

🟢 Architecture: locked (2026-06-24)
🟡 Phase 0: setup TODO (Colab notebook готовий, треба прогнати)
🔴 Phases 1-4: TODO

## Deprecated артефакти

Старі diffusion sanity-скрипти у `pipelines/_deprecated_diffusion/` — НЕ запускати, тримаються тільки для git history. Якщо знадобиться diffusion для backplate inpaint (фон без об'єкта — не потрібен label) — пишемо новий мінімальний скрипт.
