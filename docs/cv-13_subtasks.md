# CV-13 — Subtasks skeleton для Jira

> Локально в `docs/` (gitignored разом з `docs/epic.md`). Завести у Jira окремо.
> Оновлено 2026-06-25 — переписано під Blender composite pipeline (попередня diffusion-версія deprecated разом з kontentom у `datasetforge/pipelines/_deprecated_diffusion/`).

Структуровано під 4 групи: **A. Locking**, **B. Generation**, **C. Eval+Versioning**, **D. QA**.

## Group A — Locking decisions

### CV-13.1 — Затвердити фінальну таксономію (v3) ✅
- **Опис:** v3 = 10 класів (Tier 1: tank, ifv_apc, artillery, air_defense, mlrs, truck_logistics, radar_ew; Tier 2: light_vehicle, motorcycle, infantry). Спалена техніка = hard negative без боксу.
- **Acceptance:** v3 затверджена командою (DRI: Даша + військовий експерт).
- **Estimate:** 1 день. **Status:** DONE.
- **Refs:** `docs/taxonomy_v3.md`

### CV-13.2 — Затвердити гайдлайн розмітки v2 ✅
- **Опис:** Pixel thresholds (≥20 px confident, 10-19 borderline, <10 skip), обов'язкові метадані (`distance_m`, `view_angle_deg`, `hfov_deg`, `season`, `landscape`, `camera`), hard negative bank rules.
- **Acceptance:** Підтверджено DRI. Інтегровано у workflow datasetforge.
- **Estimate:** 0.5 дня. **Status:** DONE.
- **Refs:** `docs/labeling_guidelines.md`

## Group B — Generation (datasetforge — Blender composite)

### CV-13.3 — datasetforge Phase 0: BlenderProc quickstart ✅
- **Опис:** Install Blender 4.x + BlenderProc + bpycv. Прогнати `basic.py` (Suzanne) на Colab T4 → переконатись що GPU OPTIX/CUDA рендерить, HDF5 пишеться.
- **Acceptance:** non-black PNG в output. **Status:** DONE.
- **Estimate:** 0.5 дня. Бюджет: $0.
- **Refs:** `datasetforge/pipelines/colab/blender_smoke_kaggle.ipynb`

### CV-13.4 — datasetforge Phase 1: `light_vehicle` smoke (G1) ✅
- **Опис:** Engine skeleton (`scene_builder`, `bbox_extractor`, `camera_sampler`, `season_lighting`, `render_runner`) + config `v1_light_vehicle.yaml`. 3 × GAZ Tigr `.glb`, 4 сезони HDRI+ground (Poly Haven), 20 кадрів на Kaggle P100.
- **Acceptance:** 20/20 кадрів — техніка прямо, bbox tight, без втоплення в землю. **Status:** DONE (2026-06-25).
- **Estimate:** 5 днів (фактично 3 з 14 ітеративними фіксами, див. CLAUDE.md). Бюджет: $0.
- **Refs:** `datasetforge/configs/v1_light_vehicle.yaml`, `docs/datasetforge_roadmap.md` §Phase 1.

### CV-13.5 — datasetforge Phase 2: Backplate compositor
- **Опис:** Розширити `scene_builder` параметром `backplate_path`. Blender compositor: 3D vehicle render з alpha → composite над real-photo backplate (drone-shot field/forest_belt без техніки). Джерело backplates: inpaint існуючих кадрів з `data/external/sources_roboflow/` + `mendeley_uav/` через FLUX inpaint щоб прибрати техніку. Sentinel-2 fallback для відсутніх сезонів.
- **Acceptance:** 20 кадрів `light_vehicle` з real-photo backplate замість PBR ground. Reviewer A/B не відрізняє від reference real-drone footage у >50% випадків.
- **Estimate:** 1 тиждень. Бюджет: ~$5 (FLUX inpaint Colab).
- **Deps:** CV-13.4.

### CV-13.6 — datasetforge Phase 3: Scale-out 8 решти класів
- **Опис:** Клонувати `v1_light_vehicle.yaml` для кожного з 8: tank, ifv_apc, artillery, air_defense, mlrs, truck_logistics, radar_ew, motorcycle, infantry. Tier-залежний `distance_m`: дрібні (≤4м) — [150-500], великі (8-12м) — [200-1000]. 3D-моделі: Sketchfab CC-BY (9 класів безкоштовно), TurboSquid radar_ew (~$160).
- **Acceptance:** 800-1000 кадрів per class × 10 = ~9000 positives. Per-class bbox histogram median 50-80 px. `dataset/inspect.py` усі сплити pass.
- **Estimate:** 2 тижні. Бюджет: ~$20-30 (HF Jobs L4 batch) + $160 radar моделі.
- **Deps:** CV-13.5.

### CV-13.7 — Degradation pipeline
- **Опис:** Post-render: motion blur (0-8 px), JPEG compression (q60-95), atmospheric haze overlay (strength 0-0.4). Випадкові параметри per image, фіксовані per-seed. Реалізація в `datasetforge/degradation/`.
- **Acceptance:** Усі кадри пройшли degradation. Visual A/B з reference real drone footage.
- **Estimate:** Паралельно з Phase 3. Бюджет: $0 (Colab).
- **Deps:** CV-13.6.

### CV-13.8 — Hard negatives (2000 кадрів)
- **Опис:** Empty landscapes (4 сезони × 4 ландшафти) + procedural-destroyed Tier-1 (Boolean cuts + charred PBR materials, без боксу) + civilian-truck-only frames.
- **Acceptance:** 2000 hard negatives. ratio 18-22% від загального positives.
- **Estimate:** 1 тиждень. Бюджет: $0.
- **Deps:** CV-13.6.

## Group C — Eval + Versioning

### CV-13.9 — OSINT real-drone eval set (200-500 кадрів)
- **Опис:** Зібрати реальні drone кадри з Telegram OSINT (Madyar, Achilles, Birds of Magyar). **ТІЛЬКИ** для eval/test. Розмітити вручну за нашою таксономією. Legally review.
- **Acceptance:** 200-500 кадрів в `data/eval_real/` з YOLO labels + metadata sidecar.
- **Estimate:** 2-3 тижні (трудомістко через legal + manual labeling).
- **Deps:** CV-13.1.

### CV-13.10 — Train/val/test split з seed
- **Опис:** Stratified split за class × distance × season. Seed locked. `_meta/splits/` записаний.
- **Acceptance:** Кожен split покриває усі 10 класів + усі distance buckets + усі сезони. Spit lock (replay-able).
- **Estimate:** 1 день.
- **Deps:** CV-13.6, CV-13.8.

### CV-13.11 — Версіонування + пуш в HF private
- **Опис:** Пуш до `Dariachup/yolo-bluebierd-data` (приватний). Git tag `df-v1.0.0`. README з повним описом.
- **Acceptance:** Repo приватний, команда має read access. README пояснює структуру + license + статистику.
- **Estimate:** 1 день.
- **Deps:** CV-13.10.

## Group D — QA

### CV-13.12 — QA pass: per-class balance + missing labels + bbox sanity
- **Опис:** `dataset/inspect.py` + повторна валідація через скрипт. Перевірка orphan labels, дублікатів, anomaly bboxes. Sanity log з `scene_builder` (`[orient] h/L`) проти flag для лежачої техніки.
- **Acceptance:** Звіт у `data/_meta/qa_v1.md`. Усі issues класифіковані як accept/fix/reject.
- **Estimate:** 2-3 дні.
- **Deps:** CV-13.11.

### CV-13.13 — Double-pass перевірка розмітки на sample
- **Опис:** Sample 5% кадрів. Другий аннотатор перевіряє bbox tight fit, truncation rules, class assignment. IoU consistency ≥0.85.
- **Acceptance:** Sample IoU ≥0.85. Disagreement <5%.
- **Estimate:** 1 тиждень (залежить від аннотатора).
- **Deps:** CV-13.11.

## Summary

| Group | Tickets | Total estimate |
|---|---:|---|
| A. Locking | 2 | DONE |
| B. Generation | 6 (2 DONE, 4 TODO) | 3-4 тижні |
| C. Eval + Versioning | 3 | 3-4 тижні (паралельно з B) |
| D. QA | 2 | 1-2 тижні |
| **Total wall clock** | **13** | **4-6 тижнів** (від 2026-06-25) |

**Бюджет:** HF Jobs L4 batch ~$30 + radar 3D models $160 + FLUX inpaint ~$5 = **~$195**.
