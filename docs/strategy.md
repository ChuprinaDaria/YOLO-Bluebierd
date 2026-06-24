# Стратегія даних (2026-06-24, після reality-check)

## TL;DR

Public open-data warm-start = **переважно шум** для нашого реального кейсу (300-800 м, oblique, сезони, low-quality drone footage). Реалістичний шлях:

```
┌─────────────────────────────────────────────────────────────┐
│  PRIMARY:    Brave1 Dataroom (коли відкриється)             │
│              + own datasetforge synthetic під наші умови    │
│  EVAL:       OSINT-видлов реального drone footage (200-500) │
│  PRETRAIN:   public Ukraine-war + Mendeley + AMAD-5 — лише  │
│              для feature backbone, НЕ для production train  │
└─────────────────────────────────────────────────────────────┘
```

## Що змінилось

Розпаковано повний `synthetic_apc_726` (728 кадрів, 1 клас APC). Це генератор `datasetforge` (Blue Bird internal). Аналіз bbox-розмірів:

| Bucket (min-side px) | Count | % |
|---|---:|---:|
| <10 px | 0 | 0% |
| 10-20 px | 12 | 1.9% |
| 20-50 px | 203 | 31.4% |
| 50-100 px | 368 | 57.0% |
| 100-200 px | 59 | 9.1% |
| >200 px | 4 | 0.6% |

Mдіана 60 px, мін 13 px. **Це саме той distribution, що потрібен для 300-800 м oblique.** Контраст з public-сетами, де bbox-и зазвичай 200-500 px (close-up museum/parade photos).

## Що це означає

**`datasetforge` = primary warm-start engine.** Не public датасети. Public — pretrain backbone тільки.

## План — переосмислений

### Stage 1: datasetforge expansion (НАЙВАЖЛИВІШЕ)

Адаптувати internal datasetforge tool під всі наші класи:

| Клас | Зараз | Потрібно |
|---|---|---|
| apc | ✅ 726 кадрів | OK |
| tank | ❌ | 500-1000 з тим же distribution |
| artillery | ❌ | 500-1000 |
| air_defense | ❌ | 500-1000 (Pantsir, Buk, S-300/400) |
| mlrs | ❌ | 500-1000 (Grad, Smerch, Uragan) |
| truck_logistics | ❌ | 500-1000 (KamAZ, Ural) |
| radar_ew | ❌ | 300-500 |
| destroyed_vehicle | ❌ | 300-500 (vs боєздатна) |
| infantry, motorcycle, light_vehicle | ❌ | tier 2, окремий збір |

Параметри генерації мають включати:
- Висота 300/400/600/800 м (стратифіковано)
- Кут 10-60° від горизонту
- Сезон: літо/зима/багнюка/весна
- Атмосфера: чисто/туман/дощ/сніг
- Degradation pipeline: motion blur, JPEG compression artifacts, low-light, partial occlusion
- Hard negatives: порожні поля, лісосмуги, спалена техніка без box

Метадані обов'язкові — як у `docs/labeling_guidelines.md` (`altitude_m`, `view_angle_deg`, `season`, `landscape`, `camera`).

### Stage 2: OSINT eval set (важливо)

200-500 реальних drone-кадрів з Telegram OSINT (Madyar, Birds of Magyar, Achilles, NCDF). **Тільки для validation/test**, не для train (юридично сіро + може містити sensitive). Це наш `mAP_real` ground truth.

Без цього всі цифри — фікція.

### Stage 3: Pretrain backbone (опція)

З public даних що вже скачали (~68k кадрів):
- Mendeley UAV: 7,985 — bird's-eye view
- AMAD-5: 28k — aerial military  
- rawsi18: 22k — mixed military
- Roboflow Ukraine sets: ~2.5k

Дві опції:
1. **Pretrain backbone тільки** (ImageNet → military_general → наш datasetforge). NORM mAP не вимірюємо.
2. Skip взагалі. Беремо yolo11n.pt з ImageNet pretrain і йдемо одразу на datasetforge train.

### Stage 4: Brave1 (primary, коли відкриється)

Як тільки доступ — primary training там. Local code портабельний.

## Що припиняємо робити

- ❌ Priority-2 Roboflow downloads — більше шуму не треба
- ❌ Decode `A1-A20` defence-rslx9 — marginal value
- ❌ Sentinel-2 backgrounds — datasetforge має робити свої
- ❌ Synthetic compositing з public foreground — datasetforge замінює це

## Що починаємо робити

1. **Знайти `datasetforge`** — де код? Хто owner? Чи можна форкнути в `YOLO-Bluebierd` як `dataset/forge/` або тримати окремим repo з submodule?
2. **Подивитись параметри генерації APC** — скільки тюнингу для tank/artillery/AD?
3. **Inventory `datasetforge`** — які 3D-моделі техніки доступні? Чи є Pantsir/BUK/Grad/etc?
4. **Reproducibility:** запустити datasetforge локально на ноуті dg@Mac або в HF Job, отримати 500 нових кадрів tank → перевірити що pipeline працює.
5. **OSINT eval set:** окрема таска. Питання правової оцінки — обговорити з тобою.

## Питання

1. **Де datasetforge code?** GitHub `bluebird-works/datasetforge` я не бачив у списку. Це окрема repo? Чи у `cv_targeting_plane`? У `dg@Mac` локально?
2. **Хто owner datasetforge?** (бачив що `/Users/dg/Documents/datasets/ground_targets` — це `dg` Мак). Це Dmytro Gavryliuk (з SharePoint посилання)?
3. **3D-моделі техніки:** які доступні в datasetforge? Тільки APC? Чи можна докинути tank/artillery/тощо?
4. **Brave1 timeline:** 2 тижні? 2 місяці? 6+? Це визначає скільки інвестуємо в datasetforge expansion.
