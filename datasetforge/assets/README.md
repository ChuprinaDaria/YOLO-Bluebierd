# datasetforge/assets

3D-моделі, HDRI, текстури, backplate-фони для Blender composite render.

> Уся ця папка **gitignored** через .gitignore root rules (`data/`, `assets/`). Файли push-аться у приватний HF Dataset `Dariachup/yolo-bluebierd-assets` як cache-шар. Кожен render job клонує з HF при старті.

## Структура

```
assets/
├── models/                 — .blend файли per class
│   ├── light_vehicle/      Phase 1: GAZ Tigr CC-BY
│   ├── ifv_apc/            Phase 1.5: BTR-80 (Coozy "Abandoned Soviet BTR-80")
│   ├── tank/               Phase 3
│   ├── artillery/          Phase 3
│   ├── air_defense/        Phase 3
│   ├── mlrs/               Phase 3
│   ├── truck_logistics/    Phase 3
│   ├── radar_ew/           Phase 3 ($160 spend — TurboSquid)
│   ├── motorcycle/         Phase 3
│   ├── infantry/           Phase 3
│   └── destroyed_proc/     Phase 4: procedural (Boolean cuts + charred PBR)
│
├── hdri/                   — Poly Haven CC0 sky HDR
│   ├── summer/
│   ├── autumn_mud/
│   ├── winter/
│   └── spring/
│
├── textures/ground/        — Poly Haven CC0 4K PBR
│   ├── summer/             grass, dry yellow
│   ├── autumn_mud/         brown wet mud, leafless
│   ├── winter/             snow + dark soil
│   └── spring/             muddy with sprouts
│
├── backgrounds/            — real drone backplates
│   ├── polycam/            photogrammetry terrain scans (CC-BY)
│   └── inpainted/          Roboflow drone frames з видаленою технікою (FLUX-inpaint)
│
└── normalize_blend.py      — pre-import normalizer (scale/origin/textures)
```

## Provenance per file

Кожен `.blend`, `.hdr`, `.png`/`.exr` PBR має ентрі тут з URL та license.

### models/

| Path | Source URL | Author | License | Format orig | Phase |
|---|---|---|---|---|---|
| `models/light_vehicle/gaz_tigr.blend` | TBD (Sketchfab "GAZ Tigr") | TBD | CC-BY | TBD | 1 |
| `models/ifv_apc/btr_80_abandoned.blend` | https://sketchfab.com/3d-models/abandoned-soviet-btr-80-32145d6303e5487e9d92097b9845ef02 | Coozy | CC-BY | .glb | 1.5 |
| `models/tank/t72_laaskz.blend` | https://sketchfab.com/3d-models/t-72-4bcab982fdf94665819f622d2b4fb47c | Laaskz | CC-BY | TBD | 3 |
| `models/artillery/2s19_msta_s.blend` | https://sketchfab.com/3d-models/2s19-msta-s-self-propelled-artillery-c12432ada9a84859bc74c3d5ef7046f8 | Muhamad Mirza Arrafi | CC-BY | TBD | 3 |
| `models/air_defense/pantsir_s1.blend` | https://sketchfab.com/3d-models/pantsir-s1-3cb67b0bae10418190e7ce32142231e4 | SanderWolf | CC-BY | TBD | 3 |
| `models/mlrs/bm30_smerch.blend` | https://sketchfab.com/3d-models/low-poly-bm-30-smerch-dac1ee1eeae04770a3d4cdff83351d6c | SIpriv | CC-BY | TBD | 3 |
| `models/motorcycle/lowpoly_bike.blend` | https://sketchfab.com/3d-models/low-poly-motorcycle-2-9e79295e99654e2a9fa930b5139a7d84 | Sidra | CC-BY | TBD | 3 |
| `models/infantry/lowpoly_soldier.blend` | https://sketchfab.com/3d-models/low-poly-soldier-daf5de38902e458aa57ff5ba9460ca02 | Kolos Studios | CC-BY | TBD | 3 |
| `models/truck_logistics/ural_4320.blend` | TBD (Sketchfab "Ural truck") | TBD | CC-BY | TBD | 3 |
| `models/radar_ew/krasukha_4.blend` | https://www.cgtrader.com/3d-models/military/military-vehicle/krasukha-electronic-warfare-system | CGTrader vendor | Paid (~$80) | TBD | 3 |
| `models/radar_ew/kasta_2e2.blend` | TBD (TurboSquid Russian Missile Systems Collection 4) | TBD | Paid (~$80) | TBD | 3 |

### hdri/

| Path | Source URL | License |
|---|---|---|
| `hdri/summer/*.hdr` | https://polyhaven.com/hdris (filter: outdoor, sunny) | CC0 |
| `hdri/autumn_mud/*.hdr` | https://polyhaven.com/hdris (filter: overcast, drizzle) | CC0 |
| `hdri/winter/*.hdr` | https://polyhaven.com/hdris (filter: winter, low sun) | CC0 |
| `hdri/spring/*.hdr` | https://polyhaven.com/hdris (filter: overcast, light) | CC0 |

### textures/ground/

| Path | Source URL | License |
|---|---|---|
| `textures/ground/summer/grass_*` | Poly Haven 4K PBR | CC0 |
| `textures/ground/autumn_mud/mud_*` | Poly Haven 4K PBR | CC0 |
| `textures/ground/winter/snow_*` | Poly Haven 4K PBR | CC0 |
| `textures/ground/spring/fresh_grass_*` | Poly Haven 4K PBR | CC0 |

### backgrounds/

| Path | Source | License |
|---|---|---|
| `backgrounds/polycam/*` | https://poly.cam/3d-models (aerial photogrammetry) | per-asset (зазвичай CC-BY) |
| `backgrounds/inpainted/*` | extracted from `data/external/sources_roboflow/` + FLUX-inpaint to remove vehicles | mixed (CC-BY 4.0 базово, наша inpaint версія = похідне) |

## Coverage manifest

`backgrounds/manifest.yaml` (TBD коли почнемо backgrounds prep) перераховує per-season/landscape availability щоб render runner міг stratify-sample.

## Normalize step

Кожна модель з Sketchfab/CGTrader перед використанням проходить `normalize_blend.py`:
1. Scale check + fix (cm → m).
2. Origin re-center на base of vehicle.
3. Texture re-link якщо missing.
4. Z-up axis check.

Це обов'язково — Sketchfab експорти часто мають проблеми зі scale (cm not m), origin (centered to centroid not base), missing texture refs.

## Чому не git LFS

Розмір (~500 MB - 2 GB) надлишковий для git LFS квот. HF Dataset = безкоштовний приватний storage до 100 GB.
