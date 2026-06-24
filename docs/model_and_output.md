# Модель + структура видачі даних

## 1. Вибір моделі

### Варіанти

| Модель | Params | Speed (T4) | Strength | Weakness |
|---|---:|---|---|---|
| **YOLO11n** | 2.6M | ~3 ms | швидкий, edge-deployable, default ultralytics | менш точний на дрібних обʼєктах |
| **YOLO11s** | 9.4M | ~5 ms | balance | повільніше |
| **YOLO11m** | 20M | ~10 ms | вища точність | важче для edge |
| **YOLO11n-obb** | 2.7M | ~3 ms | **OBB** — наша oblique view! | потребує OBB-labels |
| **YOLO12n** (новий) | ~3M | swift | attention-mechanism | новий, менш battle-tested |
| **RT-DETR-L** | 32M | ~12 ms | DETR transformer, no NMS | важкий, train-heavy |
| **YOLO-World** | varies | varies | open-vocab | для zero-shot, не наш кейс |

### Рекомендація

**3-stage план**:

1. **Stage 1 (warm-start, MVP):** `yolo11n.pt` ImageNet pretrain → fine-tune на synthetic 12k.
   - Один прогон. Baseline mAP.
   - Швидко, дешево, кожен крок зрозумілий.

2. **Stage 2 (production):** `yolo11s.pt` ImageNet pretrain → fine-tune на synthetic 12k + warm-start public ~10k filtered.
   - Якщо `n` валиться на дрібних — переходимо на `s`.
   - Або `yolo11m` якщо edge constraint допускає 10ms.

3. **Stage 3 (oblique optimization):** `yolo11n-obb` або `yolo11s-obb` коли датасет матиме oriented boxes.
   - Тільки якщо axis-aligned bbox дає погану localization на 800m oblique cases.
   - V2 task.

### Чому не RT-DETR

- Training **5-10× повільніший** на нашому датасеті.
- Edge deployment гірший (немає edge-optimized weights).
- Бенефіт DETR (no NMS, transformer attention) — marginal на 10 класах де геометрія важливіша за semantics.

### Baseline warm-start

Маємо `train-3_best.pt` з `cv_targeting_plane` (yolo11n, mAP50 0.640 на public 7-class). Використовуємо як **starting weights** для нашого 10-class fine-tune якщо backbone features корисні. Це швидше ніж ImageNet.

## 2. Структура output dataset

### HF private repo

```
Dariachup/yolo-bluebierd-data  (private)
├── README.md                  ← описує structure, licenses, версію
├── data.yaml                  ← master YOLO config (10 cls, train/val/test paths)
├── train/
│   ├── images/                ← *.jpg 1024×1024 (downscale 640 при train)
│   └── labels/                ← *.txt YOLO bbox
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── metadata/
│   └── *.json                 ← per-frame JSON sidecar (altitude, angle, season, ...)
├── _meta/
│   ├── version.yaml           ← df-v1.0.0, generation params, seeds
│   ├── splits/                ← seed-locked split assignments
│   ├── class_balance.csv      ← per-split per-class instance counts
│   └── source_breakdown.csv   ← кільки synthetic / pretrain / OSINT real
└── _eval_real/                ← окремий розділ для OSINT eval (≤500 кадрів)
    ├── images/
    ├── labels/
    └── metadata/
```

### Master data.yaml

```yaml
# Дататасет YOLO-Bluebierd v1.0.0
path: .
train: train/images
val: valid/images
test: test/images
eval_real: _eval_real/images   # custom split для real OSINT eval

nc: 10
names:
  0: tank
  1: ifv_apc
  2: artillery
  3: air_defense
  4: mlrs
  5: truck_logistics
  6: radar_ew
  7: light_vehicle
  8: motorcycle
  9: infantry

# Не-стандартний metadata block для нашого pipeline
source:
  generator: datasetforge
  version: 1.0.0
  seed: 42
  generated_at: 2026-MM-DD
  total_images: 12000
  hard_negatives: 1800
```

### Per-frame metadata JSON sidecar

```json
{
  "image_id": "tank_summer_400m_30deg_field_0001",
  "source": "synthetic",
  "dataset_version": "df-v1.0.0",
  "seed": 42,
  "altitude_m": 400,
  "view_angle_deg": 30,
  "hfov_deg": 70,
  "sensor_res": [1920, 1080],
  "modality": "EO",
  "season": "summer",
  "landscape": "field",
  "weather": "clear",
  "time_of_day": "day",
  "degradation": {
    "jpeg_q": 85,
    "blur_px": 4,
    "atmo_strength": 0.1
  },
  "class_name": "tank",
  "has_targets": true,
  "n_boxes": 1
}
```

### Local mirror

```
data/
├── raw/                       ← warm-start (gitignored)
│   ├── _synthetic_apc_726/    ← style anchor reference
│   ├── _sources/              ← public Roboflow + Kaggle
│   ├── _meta/                 ← inventory, candidates
│   └── kaggle/
├── datasetforge_v1/           ← фінальний synth output (gitignored, push в HF)
│   ├── train, valid, test
│   └── metadata/
└── eval_real/                 ← OSINT validation set (gitignored, sensitive)
```

## 3. Версіонування

- `git tag df-v1.0.0` на release.
- HF dataset versioning через гілки: `main`, `df-v1.0.0`, `df-v1.1.0`, ...
- `_meta/version.yaml` фіксує: dataset version, datasetforge code commit, seed, total frames, hard negative ratio.

## 4. Що не робимо

- ❌ Окремий repo per клас. **Один великий repo з папками-класами всередині** (як юзер сказала).
- ❌ Mixing public public foreground у production train data. Тільки для pretrain backbone окремо.
- ❌ OBB на v1. Axis-aligned bbox як baseline.
- ❌ Multimodal (EO + IR). Тільки EO у v1, IR як V2.
