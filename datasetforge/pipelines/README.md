# datasetforge — diffusion background pipelines (як вірно використовувати)

Генерація синтетичних кадрів для дрон-детекту: Blender рендерить техніку з ідеальним
bbox, diffusion-модель домальовує реалістичний фон, а фінальний `polish` «псує» кадр під
реальну дрон-камеру. **bbox/мітки нікуди не рухаються** — техніка вставляється назад
піксель-у-піксель, а polish — лише піксельні фільтри.

> Зразок цілі — клас `light_vehicle` (GAZ Tigr), config `configs/v1_light_vehicle.yaml`.
> Усі 3 пайплайни ділять Stage 1/4/5; різниться лише модель у Stage 3.

---

## 1. Три пайплайни — коли який

| # | Notebook | Модель (Stage 3) | Ліцензія | Коли брати |
|---|----------|------------------|----------|------------|
| 1 | `runpod/notebook.ipynb` | **FLUX.1-Depth-dev** (depth-CN inpaint) | NON-commercial | Найкраща геометрія/перспектива (depth тримає структуру) |
| 2 | `runpod/notebook_qwen2509.ipynb` | **Qwen-Image-Edit-2509** (instruction edit) | **Apache-2.0 ✓** | **Прод BlueBird** — комерційно чисто; найприродніший фон |
| 3 | `runpod/notebook_fluxfill.ipynb` | **FLUX.1-Fill-dev** (mask inpaint) | NON-commercial | Найчистіший inpaint навколо техніки; тільки тест якості |

**Порада:** для A/B-тесту прожени всі три на одному кадрі (`seed=42`) і порівняй `final/`.
Для фінального датасета під продакшн — **Qwen** (єдиний з чистою ліцензією).

---

## 2. Потік — 5 стадій (спільний для всіх 3)

```
Stage 1  Blender render (render_runner.py)   → images/ depth/ normals/ vehicle_masks/ labels/ metadata/
Stage 3  модель домальовує фон               → ai_bg/      (per-pipeline: inpaint_one / edit_one / fill_one)
Stage 4  composite: relight + вставка техніки → composite/  (shared/composite.py — bbox-safe freeze)
Stage 5  polish: туман/зерно/шум/jpeg/blur    → final/      (shared/polish.py — albumentations, pixel-only)
```

- **Stage 4** узгоджує яскравість/тон техніки під фон (`relight`) і **вставляє оригінальні
  пікселі техніки** по бінарній масці — техніка не «вирізана».
- **Stage 5** накладає камерну деградацію на **весь** кадр (зшиває техніку+фон у єдину камеру).
- **Фінальний датасетний кадр — `final/{stem}.png`.** Мітки — `labels/{stem}.txt` (YOLO), не змінюються.

---

## 3. Передумови

- **GPU:** RunPod A100/H100 ≥40GB (bf16, швидко) **або** Kaggle T4/P100 16GB (повільно, через offload/gguf — див. §7).
- **`HF_TOKEN`** у змінних середовища (Settings → Secrets на Kaggle / Pod env vars на RunPod).
- Репо склонене у `/workspace/yolo-bluebierd` (шлях зашитий у ноутбуках).
- Залежності ставить сам ноутбук: `pip install -r requirements_diffusion.txt`
  (там diffusers≥0.36, albumentations, gguf/optimum-quanto).

---

## 4. Як запускати

Відкрий потрібний notebook і виконуй cell-и зверху вниз:

1. **⚙️ SETTINGS** — головний селл-конфіг (§5). Редагуєш тут, не в YAML.
2. **env check** — друкує GPU/VRAM і обраний режим precision.
3. **repo + imports** — `git pull`, pip install, диспетчеризація моделі за `SETTINGS['model']`.
4. **Stage 1 render (1 кадр)** — пише `images/depth/normals/masks/labels/metadata`.
5. **Load model** — вантажить pipeline (один раз, повільно).
6. **Stage 3** → `ai_bg/`.
7. **Stage 4+5** → `composite/` + `final/`, показує 4-up прев'ю (raw → ai_bg → composite → final+bbox).
8. **⚖️ GATE 1** — оком оціни 1 кадр. ОК → 5 кадрів (GATE 2) → 20-кадровий батч → zip → regression-аудит.

Два «гейти» (1 → 5 → 20) щоб не палити GPU-години на поганих налаштуваннях.

---

## 5. SETTINGS — шпаргалка ручок

Усе вгорі ноутбука, у словнику `SETTINGS` + текстові промпт-блоки. `apply_settings()`
зливає це у конфіг; `_diffusion_overrides()` / `_polish_overrides()` — що саме йде у Stage 3 / Stage 5.

| Ручка | Що робить | Тюнинг |
|-------|-----------|--------|
| `guidance` | сила слідування промпту | ↓ якщо «пластик»/оверсат (FLUX-Depth 3.5→2.5; FLUX-Fill ~30; Qwen 1.0) |
| `strength` | (FLUX-Depth) скільки фону переписати | ↓ 0.88→0.80 якщо втрачає aerial-масштаб |
| `steps` | кроки дифузії | ↑ 40-50 якщо фон надто blurred |
| `true_cfg_scale` | вмикає `negative_prompt` (>1) | distilled-моделі ігнорують негатив при 1.0; 2-4 = негатив активний (≈×2 час). Qwen дефолт 4.0 |
| `lora_*` | realism-LoRA (FLUX #1/#3) | найсильніший анти-«пластик»; `lora_scale` 0.6-1.0 |
| `relight_*` | підтягує яскравість/тон техніки під фон | `relight_strength` 0.4-0.7; `match_color` per-channel WB |
| `POLISH` | Stage 5 камерна деградація | §6 |
| `road_under_vehicle` | колія під технікою у Blender | `true` проти «техніка посеред поля» |
| `depth_control_*` | (Qwen) depth-ControlNet | дефолт `False` — diffusers-API нестабільне |
| `force_precision` / `gguf_url` | VRAM-режим | §7 |

**Три режими ітерації:**
- Змінив `guidance/strength/steps/relight/polish/промпт/true_cfg_scale` → re-run **SETTINGS → Stage 3 → Stage 4/5** (без reload, без re-render — швидко; SETTINGS оновлює `diff_cfg`/`polish_cfg` in-place).
- Змінив `lora_*` / `force_precision` / `base_model` / `gguf_url` → re-run **Load-cell** (модель перевантажується).
- Змінив `road_under_vehicle` / `landscapes` / scene → re-run **Stage 1 render** і далі.

---

## 6. Stage 5 polish (albumentations)

`shared/polish.py` будує `A.Compose` з блоків `POLISH` і застосовує до `composite/` → `final/`.
Усі трансформи **pixel-only → bbox не рухається**; результат **seeded** (відтворюваний по кадру).

Блоки (кожен: `false` щоб вимкнути; `p` = ймовірність; решта — діапазони albumentations 2.x):
`iso_noise`, `gauss_noise` (сенсорний шум), `motion_blur`/`defocus`, `fog` (aerial димка),
`brightness_contrast`, `chromatic` (хром. аберація), `downscale`, `jpeg` (головний «дрон/телефон» tell).

- Замало «брудно/камерно» → ↑ `p` у `iso_noise`/`gauss_noise`, ↓ `jpeg.quality`.
- Надто розмито → `defocus: false`, `downscale: false`.
- Це **той самий** augmentation, що зближує датасет із реальним дроновим відео.

> Стара cv2-деградація всередині composite **видалена** — уся «зйомка під камеру» тепер тут.

---

## 7. VRAM / квантизація (Kaggle ↔ RunPod)

`shared/precision.py → select_precision()` сам обирає режим за VRAM:

| VRAM | Режим | Що відбувається |
|------|-------|-----------------|
| ≥40GB | `bf16` | повністю на GPU, швидко (RunPod) |
| ≥22GB | `fp8_offload` | bf16 + sequential CPU offload (L4/A10) |
| <22GB | `gguf_offload` | gguf-Q4 трансформер + offload (Kaggle 16GB) |

- **Kaggle 16GB:** Qwen (20B) і FLUX-Fill у bf16 **не влізуть**. Або встав `gguf_url` у SETTINGS
  (URL gguf-Q4 трансформера), або лишиться bf16 + sequential offload — **працює, але повільно
  (хвилини/кадр)**. Це безкоштовний тест якості, не throughput.
- `force_precision: 'bf16' | 'fp8_offload' | 'gguf_offload'` — ручне перевизначення авто-вибору.
- 20B Qwen + важкий Qwen2.5-VL енкодер на 16GB тісно навіть у Q4 — спершу 1 кадр, не батч.

---

## 8. Ліцензії (важливо для BlueBird)

| Модель | Ліцензія | Комерційне використання |
|--------|----------|--------------------------|
| Qwen-Image-Edit-2509 | **Apache-2.0** | ✅ вільно |
| FLUX.1-Depth-dev | FLUX.1-dev NC | ❌ тільки R&D/тест |
| FLUX.1-Fill-dev | FLUX.1-dev NC | ❌ тільки R&D/тест |

**Для продакшн-датасета BlueBird бери Qwen.** FLUX — лише як референс якості на тесті.

---

## 9. Troubleshooting

| Симптом | Причина / фікс |
|---------|----------------|
| `[lora] FAILED ...` | LoRA не сіла на цю модель — pipeline продовжує без неї. Інша realism-LoRA або `lora_enabled: False`. |
| `[gguf] FAILED ...` | поганий `gguf_url` / версія — fallback на bf16+offload. Перевір URL на HF. |
| OOM на Kaggle | bf16 не влазить → встав `gguf_url` або поклади `force_precision: 'gguf_offload'`; жени 1 кадр, не батч. |
| Фон «пластиковий» | `guidance` ↓; увімкни realism-LoRA (FLUX); Qwen — `true_cfg_scale` ↓. |
| Техніка іншої яскравості | `relight_enabled: True`, `relight_strength` ↑. |
| Техніка «посеред поля» | `road_under_vehicle: True`; у промпті landscape-cue вже кладе колію. |
| Qwen перемальовує техніку | depth-CN тут не потрібен — Stage 4 морозить техніку; підсиль інструкцію у промпті. |
| `negative_prompt` не діє | distilled-модель — підніми `true_cfg_scale` >1 (дорожче). |

---

## 10. Вихідна структура (на кадр `{stem}`)

```
output/<run>/
├── images/{stem}.jpg          Stage 1 RGB (raw 3D)
├── labels/{stem}.txt          YOLO bbox — НЕ змінюється жодною стадією
├── metadata/{stem}.json       сцена/камера/сонце/seed
├── depth/{stem}.png           16-bit depth (mm) — вхід для FLUX-Depth/Qwen-CN
├── normals/{stem}.png         16-bit 3ch — для relight Lambert-модуляції
├── vehicle_masks/{stem}.png   8-bit, техніка@255 — freeze-маска
├── ai_bg/{stem}.png           Stage 3 — фон від моделі
├── composite/{stem}.png       Stage 4 — фон + заморожена техніка
└── final/{stem}.png           Stage 5 — ★ ДАТАСЕТНИЙ КАДР (polished)
```

Для тренування бери пари **`final/` + `labels/`**.

---

## 11. Застереження (sim-to-real)

Diffusion-фон — зручно, але має ризик: модель домальовує фон зі своїм світлом/тінню/масштабом,
і YOLO може почати ловити **артефакт композитингу**, а не саму техніку. Ми це пом'якшуємо:
`relight` (узгодження світла), `road_under_vehicle` (масштаб/контекст), єдиний `polish` на весь
кадр (спільна «камера»). Але повністю проблему це не знімає.

**Альтернатива на майбутнє:** робити фон **нативно в BlenderProc** (HDRI + ground-текстури) —
тоді світло/тінь/перспектива консистентні з коробки, bbox ідеальний, а diffusion лишається як
domain-randomization добавка, не як основа. Зерно/шум у будь-якому разі — `polish` (albumentations),
не генеративна модель.
