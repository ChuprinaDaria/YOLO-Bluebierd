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

**Engine inside datasetforge (рішення 2026-06-24):** Blender 3D composite render (BlenderProc), НЕ diffusion. Причина: text2image не дає bbox labels, а вся цінність синтетики для detection полягає у тому що генератор знає де об'єкт. Composite render з 3D-меша → projection → bbox автоматом. Diffusion (FLUX/SDXL/GLIGEN/LoRA) залишається тільки для inpaint фонів (бо для backgrounds labels не потрібні). Детально — `datasetforge/README.md` і план `~/.claude/plans/abundant-snacking-thacker.md`.

## План — переосмислений

### Stage 1: datasetforge build-out (статус 2026-06-25)

Власна Blender composite pipeline під 10 класів (`docs/taxonomy_v3.md`):

| Клас | Статус | Кадрів target |
|---|---|---|
| apc (reference style anchor) | ✅ 726 кадрів `_synthetic_apc_726` | OK |
| light_vehicle | ✅ Phase 1 PASS (20 smoke), reference setup | 800-1000 у Phase 3 |
| tank | TODO Phase 3 | 800-1000 |
| ifv_apc | TODO Phase 3 | 800-1000 |
| artillery | TODO Phase 3 | 800-1000 |
| air_defense (Pantsir, Buk, S-300/400) | TODO Phase 3 | 800-1000 |
| mlrs (Grad, Smerch, Uragan) | TODO Phase 3 | 800-1000 |
| truck_logistics (KamAZ, Ural) | TODO Phase 3 | 800-1000 |
| radar_ew (Krasukha, Kasta) | TODO Phase 3, $160 TurboSquid | 600-800 |
| motorcycle, infantry | TODO Phase 3, tier 2 | 600-800 each |

Параметри генерації (фіксовані у `v1_light_vehicle.yaml` як reference):
- `distance_m` 150-1000 (tier-залежно), `view_angle_deg` 10-30° oblique + 90° nadir
- `hfov_deg` 15° ISR telephoto (буде уточнено коли клієнт надасть spec реальної камери)
- Сезон: літо/осінь_багнюка/зима/весна (HDRI + PBR ground з Poly Haven)
- Lighting: HDRI strength=2.0 + explicit SUN energy=5.0
- Degradation: motion blur, JPEG compression, atmosphere haze (CV-13.7)
- Hard negatives 18-22%: empty landscapes + procedural-destroyed (Boolean cuts) + civilian-only

Метадані обов'язкові — `distance_m`, `view_angle_deg`, `hfov_deg`, `season`, `landscape`, `model_variant`, `hdri`, `ground_texture` (див. `docs/labeling_guidelines.md`).

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

## Що робимо далі (2026-06-25)

1. **Phase 2 backplate compositor** (CV-13.5) — заміна PBR ground на real-drone backplate через FLUX inpaint.
2. **Phase 3 scale-out 8 класів** (CV-13.6) — клонування `v1_light_vehicle.yaml` під tank/ifv_apc/artillery/AD/MLRS/truck/radar/motorcycle/infantry. Купівля radar_ew моделей (~$160 TurboSquid).
3. **Phase 4 hard negatives + train smoke YOLOv8n** (CV-13.7, CV-13.8).
4. **OSINT eval set** (CV-13.9) — паралельно з Phase 3. Legal review перед збором.

## Закриті питання

- ✅ **datasetforge code** — будуємо власну Blender composite pipeline у цьому repo (`datasetforge/`). Reference style anchor — `_synthetic_apc_726` від Дмитра Гавриленка (SharePoint).
- ✅ **3D-моделі техніки** — Sketchfab CC-BY (9 класів безкоштовно) + TurboSquid radar_ew ($160 разово).
- ⏳ **Brave1 timeline** — невідомо. Паралельний трек, не блокує warm-start. Code портабельний.
