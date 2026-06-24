# datasetforge roadmap

## Чому пишемо самі

`synthetic_apc_726` згенерував `dg@Mac` колись через невідомий нам tool (можливо їх власний прототип). Production-grade engine ми ще не маємо. Треба написати свій — це не optional модуль, це фундамент warm-start даних.

## Фази

### Phase 0 — Foundation (тиждень 1-2)

- ✅ Архітектура: задизайнена в `datasetforge/README.md`
- ✅ Skeleton: `engine/`, `output/metadata.py`, configs (порожні класи з NotImplementedError)
- 🔴 **Blender install + headless verification** на машині розробника (НЕ на Phenom II — без SIMD рендер впаде). HF Jobs з GPU чи окремий dev-вузол.
- 🔴 Один end-to-end "Hello World" рендер: куб на полі з камерою на 300м.

### Phase 1 — MVP single-class (тиждень 3-4)

- 🔴 Імпорт першої 3D-моделі (наприклад tank: T-72B3 з TurboSquid/CGTrader).
- 🔴 Камера-rig: altitude/angle/HFOV з конфігу.
- 🔴 Bbox extractor: 3D bound_box → 2D YOLO bbox через `bpy_extras.world_to_camera_view`.
- 🔴 Output: YOLO label + metadata JSON.
- 🔴 Сценарій: 100 кадрів tank, 1 ландшафт (field), 1 сезон (summer) — sanity.
- 🔴 Інспекція через `dataset/inspect.py` → bbox distribution має матчити `synthetic_apc_726` стилю (мediana ~60 px, мін ≥13).

### Phase 2 — Multi-condition (тиждень 5-6)

- 🔴 Backgrounds asset library: OpenAerialMap + VisDrone + OSINT перші 100 frames.
- 🔴 Сезонні текстури: ground PBR за seasons (literature: green grass, mud, snow, bare soil).
- 🔴 Lighting: sun angle відповідно до time_of_day + season.
- 🔴 Degradation pipeline: motion_blur, jpeg, atmosphere overlay.
- 🔴 Hard negatives: 15% ratio, background-only + civilian context.
- 🔴 Сценарій: 1000 кадрів tank, повна стратифікація.
- 🔴 Train baseline yolo11n on this — sanity check that mAP > random.

### Phase 3 — Multi-class scale (тиждень 7-10)

- 🔴 Імпорт усіх 10 класів (3D-моделей).
- 🔴 Batch orchestrator з parallelism (HF Job per class).
- 🔴 Конфіги per-class: `configs/v1_<class>.yaml`.
- 🔴 Full datasetforge v1.0.0 release: 12,000 кадрів total (стратифіковано як у `taxonomy_v3.md`).
- 🔴 Push до приватного HF repo `Dariachup/yolo-bluebierd-data` з тегом `df-v1.0.0`.

### Phase 4 — Iteration (постійно)

- Залежно від evaluation на OSINT real eval set:
  - Якщо `mAP_real` <<< `mAP_synthetic` → domain gap. Тюнинг degradation і backgrounds.
  - Якщо false positive на лісосмугах → mine більше shelterbelt hard negatives.
  - Якщо winter < summer recall → більше snow renders.

## Reality-check: скільки часу і ресурсів

| Етап | Time | Computе | $$ |
|---|---|---|---|
| Phase 0 | 1-2 тиж | local CPU | 0 |
| 3D-моделі (Phase 1) | 1-2 тиж | — | $200-$1000 (CGTrader licenses, ~10-20 моделей) |
| Phase 1 рендер | days | local або 1× HF GPU | $5-20 на 1k кадрів |
| Phase 2-3 | 4-6 тиж | HF Jobs з GPU | $50-200 на 12k кадрів |
| Backgrounds collection | parallel | — | 0 ($) + час |

**Загалом ~2-3 місяці до stable v1.0.0** на одного програміста. З Brave1 access швидше — bypass більшої частини roadmap.

## Альтернатива: купити готовий синтетик-як-сервіс

- Datagen.tech
- Synthesis AI
- Mostly AI

Для military — рідко offer. Власний шлях більш реалістичний.

## Що паралельно НЕ блокується datasetforge

- `training/`, `evaluation/`, `inference/`, `aim_assist/` модулі — пишемо архітектуру і workflow.
- Pretrain на public ~68k — баг-зайнятість для backbone, низький пріоритет.
- OSINT eval collection — окрема активність.
