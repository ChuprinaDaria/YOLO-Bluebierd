# datasetforge

Власний синтетичний генератор кадрів під наші 10 класів цілей.

> Заміняє placeholder з `synthetic_apc_726`. Той був single-class, kept як референс стилю.
> Тут — multi-class engine під CV-13 acceptance criteria.

## Цільові характеристики (з аналізу `synthetic_apc_726`)

| | Цільове |
|---|---|
| Image size | 640×640 (resize при тренуванні) |
| Bbox min-side px | ≥10 (детекція), ≥20 (класифікація) |
| Bbox median | ~60 px |
| Bbox range | 13-220 px |
| Hard negative ratio | ~10-20% (порожні + спалена техніка без боксу) |
| Стратифікація | висота × кут × сезон × ландшафт |
| Метадані | JSON sidecar обовʼязковий |

## Стек

**Blender + Python API (bpy)** — headless, безкоштовний, AGPL, відмінний для синтетичних датасетів.

Альтернативи розглянуто:
- Unity Perception SDK — потребує commercial Unity license, перешкода для defense miltech.
- NVIDIA Omniverse Replicator — топ якість, але важкий + RTX-залежний + commercial.
- **Blender — best fit** для нашого бюджету і портабельності (Brave1 environment теж).

## Архітектура

```
config.yaml (клас, висота, кут, сезон, ...)
   │
   ▼
┌──────────────────────┐
│ engine/              │  Blender wrapper: scene assembly, camera, lighting
│   scene_builder.py   │  
│   render.py          │  Headless render → PNG
│   bbox_extractor.py  │  3D bbox → YOLO 2D bbox auto-projection
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ degradation/         │  motion_blur, jpeg_artifact, atmosphere
│   blur.py            │  (matched to drone footage realism)
│   compress.py        │  
│   atmosphere.py      │  fog/haze/rain/snow overlay
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ output/              │  YOLO label writer + JSON metadata
│   yolo_writer.py     │  
│   metadata.py        │  altitude_m, view_angle_deg, season, landscape, ...
└──────────────────────┘
```

## Папки

| | Призначення |
|---|---|
| `engine/` | Blender scene builder, camera, render, 3D→2D bbox |
| `assets/models/` | 3D-моделі техніки (.blend/.glb) — **gitignored** (великі бінарники) |
| `assets/backgrounds/` | HDR/панорами + drone-footage frames для backgrounds — **gitignored** |
| `assets/textures/` | Сезонні текстури ґрунту, рослинність, сніг — **gitignored** |
| `configs/` | YAML пресети генерації (per-class, per-scenario) |
| `pipelines/` | High-level orchestrators (generate_class.py, batch_runs.py) |
| `degradation/` | Pipeline постобробки до drone-realism |
| `output/` | YOLO writer + metadata. Output йде в `data/datasetforge_v<N>/` (поза цією папкою) |
| `tests/` | Unit tests на bbox math, config valid, output формат |

## 3D-моделі — план набору

| Клас | Потрібні моделі | Джерело-кандидат |
|---|---|---|
| `tank` | T-72B3, T-80, T-90 | купити на TurboSquid/CGTrader (mil-tier), або власна моделізація |
| `ifv_apc` | BMP-1/2/3, BTR-80/82, MT-LB | те саме |
| `artillery` | 2S19, 2S3, D-30 | те саме |
| `air_defense` | Pantsir-S1, BUK-M2, Tor-M2 | те саме |
| `mlrs` | BM-21 Grad, BM-27, TOS-1 | те саме |
| `truck_logistics` | KamAZ-5350, Ural-4320 | civilian models OK |
| `radar_ew` | Kasta-2E2, Krasukha-4 | гірше з доступністю — можливо власна моделізація |
| `light_vehicle` | Tigr, GAZ Sobol, Lada | civilian models OK |
| `motorcycle` | military + civilian | звичайні assets |
| `infantry` | rigged humanoid + uniform variations | Mixamo + custom textures |

**Burn/destroyed variants** — derivative shader applied to base models (charred PBR texture, missing parts via boolean cut).

## Бeкграунди

**Не nadir satellite.** Геометрично — потрібен oblique drone view.

Джерела:
1. **OpenAerialMap** — oblique drone frames (sparse).
2. **VisDrone** — oblique drone, Chinese terrain — useful for compositing.
3. **OSINT real drone footage** (Telegram OSINT) — perspective-correct.
4. **Sentinel-2 + perspective warp** — для seasonal texture коли інших нема.

`assets/backgrounds/` структуровано: `<season>/<landscape>/<source>/*.jpg`.

## Workflow per-class

```bash
# 1. Згенерувати конкретний клас
python -m datasetforge.pipelines.generate_class --class tank --count 1000

# 2. Згенерувати всі 10 класів за конфігом
python -m datasetforge.pipelines.batch_runs --config configs/v1_full.yaml

# 3. Перевірити output
python dataset/inspect.py data/datasetforge_v1/
```

## Reproducibility

- Все генерується з `seed` у конфізі.
- Версія datasetforge: `datasetforge.__version__` пишеться в metadata кожного кадру.
- Git tag на release: `df-v1.0.0`, `df-v1.1.0`.

## Стан

🟢 Архітектура: задизайнено
🔴 Engine: скелет (TODO)
🔴 3D-моделі: жодної (TODO acquisition)
🔴 Backgrounds: жодного (TODO збір)
🔴 Degradation: TODO

Це чергова велика частина проєкту. Дивись окремий roadmap у `docs/datasetforge_roadmap.md`.
