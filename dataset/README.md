# dataset/

Скрипти для аудиту, валідації і пушу даних у приватний HF Hub repo.

## Стратегія розміщення

**Один приватний HF repo:** `Dariachup/yolo-bluebierd-data` (dataset type, `private=True`).

Кожен клас — окрема папка верхнього рівня. Усередині — YOLO export структура (train/valid/test з images/ + labels/ + data.yaml).

```
Dariachup/yolo-bluebierd-data/                 (private)
├── README.md                                  ← глобальний опис, мапінг class_id
├── _meta/
│   ├── taxonomy.yaml                          ← фінальна таксономія
│   ├── class_mapping.yaml                     ← per-folder → global_class_id
│   └── splits/                                ← train/val/test seed-locked
├── apc/                                       ← перший клас
│   ├── data.yaml          (nc:1, names:[apc])
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── test/{images,labels}/
├── tank/
├── artillery_sp/
├── mlrs/
└── ...
```

**Чому так:**
- Кожен per-class export з Roboflow/CVAT можна додати незалежно.
- Фінальна модель = multi-class. Master `data.yaml` для тренування генерується скриптом, що зливає всі папки + ремапує `class_id` через `_meta/class_mapping.yaml`.
- Hard negatives (спалена техніка, цивільні) — окрема папка `_negatives/` з порожніми labels.
- Real vs synthetic трекаємо через `_meta/splits/source.json` для stratified eval.

## Команди

```bash
# Audit recovered dataset (single per-class folder)
python dataset/inspect.py /path/to/class_folder

# Upload one class folder to HF private repo
python dataset/upload_class.py /path/to/local_folder --class-name apc

# Generate master training data.yaml from HF repo
python dataset/build_train_yaml.py --taxonomy _meta/taxonomy.yaml
```

## Конвенції

- Класи pinned лише в `_meta/taxonomy.yaml`. Не перейменовувати ID заднім числом без міграції.
- Roboflow назви файлів (`*.rf.HASH.jpg`) залишаємо — це stable identity.
- Метадані кадру (source/season/landscape) — в JSON sidecar поряд з image: `image.json`.
