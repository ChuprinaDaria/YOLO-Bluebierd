# Можливі класи на основі усіх скачаних даних

> Аналіз: які класи реально присутні у warm-start даних + які треба добити через datasetforge.
> Висновок: v3 таксономія (10 класів) підтверджена evidence-ом. Деталі нижче.

## Усі unique класи з реальних data.yaml

Розбили на категорії за смислом. Числа в дужках — instances в train splits.

### 🟢 Heavy armor (танк / IFV / APC / MTLB) — БАГАТО

| Концепт | Знайдено в data | Total inst (train) |
|---|---|---:|
| Танк | `t-64` (94), `t-72` (93+269), `t-80` (83), `tank` (140), `military_tank` (17,432+17,243) | **~35,000+** |
| IFV / БМП | `bmp-1` (109), `bmp-2` (80), `bmp-3` (4), `bmd-2` (80) | ~280 |
| APC / БТР | `btr-70` (71), `btr-80` (86), `armoured_personnel_carrier` (161) | ~320 |
| MT-LB | `mt-lb` (87+22+62), `mtlb` | ~170 |
| Generic | `military_vehicle` (1963+10,957) | ~13,000 |

**Висновок:** з warm-start data вдосталь для `tank` + `ifv_apc`. Розрізнення T-72 vs T-80 vs T-90 НЕ критичне на 300-800м oblique (геометрично нерозрізнювані). Залишаємо як **`tank`** flat.

### 🟡 Artillery — МАЛО

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Generic artillery | `military_artillery` (439) | 439 |
| Specific тypes (D-30, 2S19, 2S3) | — | 0 |

**Висновок:** Дуже бідно для `artillery`. Треба datasetforge bulk-генерувати.

### 🟡 MLRS — ДУЖЕ МАЛО

| Концепт | Знайдено | Total inst |
|---|---|---:|
| BM-21 Grad | `bm-21` (109) | 109 |
| BM-27, BM-30, Smerch, TOS | — | 0 |

**Висновок:** Тільки Grad. Треба add Uragan/Smerch/TOS через datasetforge.

### 🟡 Air defense — encoded, незрозуміло

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Anonymous AD systems | `A1`-`A20+` (defence-rslx9, 22 cls, 14,800+ instances) | 14,800+ |
| Specific (Pantsir, BUK, Tor, S-300/400) | — | 0 |

**Висновок:** Encoded — без знання мапінгу `A1 → Pantsir` etc, marginal value. Треба decode (запит до автора defence-rslx9 на universe.roboflow.com) АБО згенерувати через datasetforge напряму з prompts.

### 🟢 Trucks — ОК

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Military truck | `military_truck` (1245), `undefined_vehicle` (56) | ~1300 |
| Civilian truck — буде як negative | — | — |

**Висновок:** ОК для `truck_logistics`. Треба фільтр military vs civilian.

### 🔴 Radar / EW — НУЛЬ

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Radar | — | 0 |
| EW (Krasukha, Murmansk, Repellent) | — | 0 |

**Висновок:** Жодного. Треба з нуля через datasetforge.

### 🟢 Infantry — БАГАТО

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Soldier | `soldier` (4474+6502+560+12,728), `camouflage_soldier` (4477) | **~28,000+** |
| People | `people` — Mendeley (Mendeley dataset specific) | ~4400 |
| (Мапінг — групи ≥3) | — | — |

**Висновок:** Багато. Але potentially mix цивільних + soldier. Treba фільтр на групи vs одинаки.

### 🟢 Civilian / light vehicle

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Civilian | `civilian` (52) | 52 |
| Civilian vehicle | `civilian_vehicle` (519+6213) | ~6,700 |

**Висновок:** OK для `light_vehicle` baseline + hard negatives.

### 🔴 Motorcycle / quad — НУЛЬ

| Концепт | Знайдено | Total inst |
|---|---|---:|
| Motorcycle | — | 0 |

**Висновок:** Жодного. Треба datasetforge.

### ❌ Не наша таксономія (skip)

| Концепт | Куди дівати |
|---|---|
| `drone` (Mendeley) | air-to-air, окрема задача V2, skip |
| `air-fighter`, `bomber`, `military_aircraft` | air-to-air, skip |
| `military_warship` (rawsi18, 2134) | maritime, skip |
| `weapon` (rawsi18, 1210) | окремий клас (зброя в руках), skip для V1 |
| `trench` (rawsi18, 4) | landscape/ background, skip |
| `missile` (різні джерела) | окрема V2 |

## Фінальний список 10 класів (підтверджений)

| ID | Code | Evidence rating | datasetforge priority |
|---:|---|---|---|
| 0 | `tank` | 🟢🟢🟢 надлишок | low — baseline 1000 |
| 1 | `ifv_apc` | 🟢🟢 норм | low — baseline 1000 |
| 2 | `artillery` | 🟡 мало (439) | **HIGH** — 1200 |
| 3 | `air_defense` | 🟡 encoded (14800) | **HIGH** — 1200 |
| 4 | `mlrs` | 🟡 дуже мало (109) | **HIGH** — 1200 |
| 5 | `truck_logistics` | 🟢 ОК (1300) | normal — 1200 (epic priority) |
| 6 | `radar_ew` | 🔴 нуль | **HIGH** — 800 (нижче бо мало в реальності) |
| 7 | `light_vehicle` | 🟢 (6700) | low — baseline 600 |
| 8 | `motorcycle` | 🔴 нуль | **HIGH** — 600 |
| 9 | `infantry` | 🟢🟢🟢 (28000+) | normal — 1200 (epic priority) |

## Дві поправки до v3 (мінор)

1. **`tank` без розділення T-72/T-80/T-90.** На 300-800м oblique вони нерозрізнювані. Flat one class.
2. **`infantry` — групи ≥3.** Поодинокі цивільні = hard negative. Соldier vs people розрізняти не будемо.

## Підтверджено

10 класів. Pivot не треба. v3 готова до lock.
