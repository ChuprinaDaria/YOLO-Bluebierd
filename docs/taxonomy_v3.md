# Таксономія v3 — фінал для datasetforge

> v3 заміняє v2. Заточена під реальний продуктовий кейс + datasetforge генерацію.

## 10 класів (фінал, відповідає CV-13 epic "~10 класів")

### Tier 1 — велика техніка (виживає на 800 м з зумом, до ~400 м широким)

| ID | Code | UA | Приклади (3D-модель потрібна) |
|---:|---|---|---|
| 0 | `tank` | Танк | T-72B, T-72B3, T-80, T-90 |
| 1 | `ifv_apc` | ББМ/БТР | BMP-1/2/3, BTR-80/82, MT-LB, BMD-2/4 |
| 2 | `artillery` | Артилерія (towed+SP, обʼєднано) | 2S19, 2S3, 2S1, D-30, 2A65 Msta-B |
| 3 | `air_defense` | ППО | Pantsir-S1, BUK-M1/2/3, Tor-M2, S-300, S-400, ZSU-23-4 |
| 4 | `mlrs` | РСЗВ | BM-21 Grad, BM-27 Uragan, BM-30 Smerch, TOS-1 |
| 5 | `truck_logistics` | Військовий транспорт | KamAZ-5350, Ural-4320, MT-LB-transport |
| 6 | `radar_ew` | Радари / РЕБ | Kasta-2E2, Krasukha-2/4, Murmansk-BN, Repellent-1 |

### Tier 2 — мала/легка (тільки нижні висоти або зум)

| ID | Code | UA | Приклади |
|---:|---|---|---|
| 7 | `light_vehicle` | Легкові і цивільні машини | Tigr, GAZ Sobol, Lada, цивільні авто, mil-pickup |
| 8 | `motorcycle` | Мотоцикли і квадри | military bikes, civilian bikes |
| 9 | `infantry` | Піхота (групи ≥3) | військові групи в полі/окопах |

## Hard negatives (БЕЗ боксу, СПЕЦІАЛЬНО невідмічені)

За CV-13 epic — це shum для калібрування:

- **Спалена/знищена техніка** будь-якого з 0-6 типів — у стані вигоріла/перевернута/без башти/без гусениць. Кадр без боксу. Це критично щоб модель не давала false positive на wrecks.
- **Цивільний транспорт у звичайному контексті** (місто, дорога).
- **Порожні ландшафти** — поле, посадка, ліс, місто без техніки.
- **Поодинокі цивільні** (1-2 людини) — щоб не плутати з `infantry`.

## Що datasetforge генерує — параметри для всіх класів

Базується на стилі `synthetic_apc_726`. Той же distribution, але масштабуємо до всіх 11 класів.

### Cтратифікація (обовʼязкова для кожного класу)

| Вимір | Значення |
|---|---|
| Висота | 300, 400, 600, 800 м — рівномірно |
| Кут | 10°, 20°, 30°, 45°, 60° від горизонту |
| Сезон | літо (зелень), осінь-багнюка, зима (сніг), весна |
| Час доби | день, сутінки (нічну виключаємо для V1 — окрема задача EO/IR) |
| Атмосфера | чисто (50%), туман (15%), дощ (15%), пил (10%), імла (10%) |
| Ландшафт | поле, посадка (лісосмуга), ліс, багнюка, місто, дорога (ґрунт+асфальт) |
| Degradation | JPEG q=60-90 (mix), motion blur, low-light, compression artifacts |

### Обʼєм на клас (відповідає epic "збалансувати датасет")

| Категорія | Кадри | Пріоритет (з epic) |
|---|---:|---|
| `air_defense` | 1200 | ⬆️ priority "більше ППО" |
| `artillery` | 1200 | ⬆️ priority "більше артсистем" |
| `mlrs` | 1200 | ⬆️ priority "більше СЗВ" |
| `truck_logistics` | 1200 | ⬆️ priority "більше вантажівок" |
| `infantry` | 1200 | ⬆️ priority "більше скупчень людей" |
| `tank` | 1000 | baseline |
| `ifv_apc` | 1000 | baseline |
| `radar_ew` | 800 | baseline |
| `light_vehicle` | 600 | tier 2 baseline |
| `motorcycle` | 600 | tier 2 baseline |
| **Subtotal classes** | **10,000** | |
| Hard negatives | 2000 | ⬆️ "фото без військової техніки" |
| **Total target** | **~12,000** | |

Це 3-5× більше **корисного сигналу** ніж public датасети сумарно.

## Per-class feedstock (що з реальних даних дає референс для datasetforge)

Це не train data — це **матеріал для тюнингу 3D-моделей і кутів** при генерації.

| Клас | Reference з public | 3D-моделі потрібні | Стан |
|---|---|---|---|
| `tank` | capstoneproject (t-64/72/80, 270 inst), tank-s4xwz (t-72 dominant, 336), MVR (140), AMAD-5 (17k) | T-72B3, T-80, T-90 | 🟢 багато refs |
| `ifv_apc` | capstoneproject (bmd-2, bmp-1/2, btr-70/80, mt-lb) — ~400, tank-s4xwz (118), MVR (161) | BMP-1/2/3, BTR-80/82, MT-LB | 🟢 |
| `artillery` | rawsi18 (439) | 2S19 Msta-S, 2S3 Akatsiya, D-30 | 🟡 малий ref-пул, потрібно докинути |
| `air_defense` | defence-rslx9 (5400, **encoded A1-A20** — треба decode) | Pantsir-S1, BUK-M2, Tor-M2 | 🟡 чи годиться без decode? |
| `mlrs` | capstoneproject (bm-21, ~100) | BM-21 Grad, BM-27, BM-30 | 🟡 малий ref |
| `truck_logistics` | rawsi18 (1245), MVR (`undefined_vehicle` 56), tank-s4xwz | KamAZ-5350, Ural-4320 | 🟡 потрібен mil/civilian filter |
| `radar_ew` | **жодного** | Kasta-2E2, Krasukha-4 | 🔴 з нуля шукати refs |
| `light_vehicle` | rawsi18 (`civilian_vehicle` 519), AMAD-5 (6213) | Tigr, GAZ, Lada | 🟢 |
| `motorcycle` | **жодного** | military bike + civilian bike | 🔴 з нуля |
| `infantry` | Mendeley (people 4474 + soldier 2983), MVR (140), AMAD-5 (12728 soldier) | прості людські моделі | 🟢 |

### Hard negative bank (окремо від 10 класів)

| Тип | Reference з public | datasetforge задача |
|---|---|---|
| Спалена/знищена техніка | rawsi18 (можливо у military_vehicle), Oryx (Kaggle piterfm) — ground perspective | згенерувати **burned versions** 3D-моделей 0-6 (charred texture, missing turret, broken tracks) — без боксу в YOLO label file |
| Цивільні авто в місті | rawsi18 (civilian_vehicle), AMAD-5, VEDAI | звичайні civilian scenes без боксу |
| Порожні ландшафти | datasetforge background-only render | hard negative kadрi без техніки |
| Поодинокі цивільні (1-2) | AMAD-5 (civilian class) | без боксу, щоб не плутати з infantry |

**Цільовий обʼєм hard negatives:** 20% від загального датасета (≈2000 кадрів). Це посилює epic вимогу "Додати фото без військової техніки".

## Що треба від `dg` / Blue Bird (datasetforge owner)

1. **Інвентар 3D-моделей** які вже є в datasetforge крім APC. Скоріш за все APC = BMP/BTR-like.
2. **Список бракуючих моделей** — особливо радари, ППО, MLRS, destroyed variants, мотоцикли.
3. **Конфіг рендера synthetic_apc_726** — скільки часу займає згенерувати 1000 кадрів? Які параметри стратифікації вже в pipeline?
4. **Доступ до коду datasetforge** — git repo посилання + credentials.

## Що ми робимо паралельно (поки чекаємо datasetforge access)

1. **Pretrain backbone** з public ~68k — okay для feature warm-start. Збиваємо в `data/train/pretrain/` через `dataset/build_unified.py` (буде).
2. **OSINT eval set** — окрема активність. Збираємо 200-500 реальних drone кадрів.
3. **Reference photos** для бракуючих 3D-моделей — збираємо посилання/фото для `radar_ew`, `motorcycle`, `destroyed`.
4. **Lock pipeline architecture** — training/, evaluation/, aim_assist/ можемо писати на yolo11n + ImageNet pretrain без чекання datasetforge.

## Decision: open датасети — pretrain only

Не зливаємо їх в одну "training" таблицю з власною таксономією. Замість:
- `data/raw/_sources/...` — як є, не чіпаємо
- `data/train/pretrain/` (буде) — unified COCO-like з 5-7 generic super-classes (`heavy_armor, light_armor, artillery_any, support_vehicle, personnel, aircraft, ship`) для backbone warm-up
- `data/train/datasetforge_v1/` — справжній train data з нашою таксономією 11 cls

Це уникне корупції фінальних class IDs.
