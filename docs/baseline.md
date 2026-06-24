# Baseline моделі (з `bluebird-works/cv_targeting_plane`)

Попередня версія задачі (Jupyter scratchpad). Беремо ваги і hyperparams як стартову точку, замінятимемо після фіналізації власної таксономії.

## Що взято

| Файл | Звідки | Розмір |
|---|---|---:|
| `data/baseline/train-3_best.pt` | `runs/detect/train-3/weights/best.pt` | 5.5 MB |
| `data/baseline/train-5_best.pt` | `runs/detect/train-5/weights/best.pt` | 5.5 MB |
| `configs/reference_args/train-3_args.yaml` | hyperparams train-3 | 1.6 KB |
| `configs/reference_args/train-3_results.csv` | per-epoch метрики train-3 | 300 рядків |
| `configs/reference_args/train-5_args.yaml` | hyperparams train-5 | 1.6 KB |
| `configs/reference_args/train-5_results.csv` | per-epoch метрики train-5 | 300 рядків |
| `configs/reference_args/train-10_args.yaml` | hyperparams train-10 | 1.6 KB |
| `configs/reference_args/train-10_results.csv` | per-epoch метрики train-10 | 300 рядків |

Ваги **під .gitignore** (`data/`). Configs і results — трекаються як reference.

## Метрики (epoch 300, val split)

| Run | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| **train-3 (best)** | yolo11n | **0.693** | **0.608** | **0.640** | **0.383** |
| train-5 (notebook default) | yolo11n | 0.598 | 0.528 | 0.556 | 0.307 |
| train-10 | yolo11n | 0.478 | 0.486 | 0.476 | 0.269 |

## Класи (7, з public Roboflow Universe)

`Artilary` (sic), `M- Rocket Launcher`, `Missile`, `Radar`, `Soldier`, `Tank`, `Vehicle`

Назви з опечатками — це джерельний датасет `nourelnouda/military-object-detection-qemer/v1`, CC BY 4.0.

## Hyperparams train-3 (winner)

- model: `yolo11n.pt`
- epochs: 300, batch: 16, imgsz: 640
- optimizer: AdamW, lr0: 0.01, lrf: 0.01, weight_decay: 0.0005
- warmup_epochs: 3
- augmentations:
  - degrees: 45, shear: 5, translate: 0.1, scale: 0.5
  - mixup: 0.2, mosaic: 1.0, fliplr: 0.5
  - hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4
  - randaugment + erasing: 0.4

## Як використовувати

```python
from ultralytics import YOLO

# Inference на новому кадрі
model = YOLO("data/baseline/train-3_best.pt")
results = model("path/to/image.jpg", conf=0.25)

# Transfer learning для нашої таксономії (коли визначимось)
model = YOLO("data/baseline/train-3_best.pt")
model.train(
    data="configs/our_taxonomy.yaml",   # коли буде
    epochs=100,
    # ... див. train-3_args.yaml як reference
)
```

## Обмеження

- Класи **не наші** — це public benchmark з англомовними опечатками. Ваги служать як warm-start, не як фінал.
- mAP50 0.64 на val — посередньо. Очікуємо суттєвого покращення на власному, чистішому датасеті.
- Train на ~GPU, локально (Phenom II без AVX) інференс працює, train — ні.
