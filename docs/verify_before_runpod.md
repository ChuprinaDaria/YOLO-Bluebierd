# Тестові прогони + візуальна верифікація ПЕРЕД RunPod

Мета: за 5-15 хвилин на дешевому/локальному залізі переконатися, що генерація
**правильна** (техніка НЕ «крапки», оклюжн реальний, bbox amodal, wreck/HN
коректні) — і лише тоді палити GPU-години на RunPod.

Стратегія: **Stage 1 (Blender-only) не потребує ні Flux, ні GPU, ні RunPod.**
Саме він містить усі критичні зміни камери/оклюжна/bbox. Спочатку ганяємо і
дивимось його, і тільки якщо він чистий — вмикаємо diffusion і йдемо на RunPod.

```
[0] unit-тести (сек)      →  [1] Blender smoke (хв)  →  [2] preview + checklist
        │                            локально/CPU              очима
        └─ логіка без Blender                                      │
                                                                   ▼
                                       усе ✓ → [3] diffusion smoke → [4] RunPod full
```

---

## 0. Швидка перевірка логіки (без Blender, секунди)

Ловить регресії у геометрії камери, фільтрі pixel-budget, amodal-bbox, OBB,
розміщенні оклюдерів — **до** будь-якого рендера.

```bash
cd <repo>
python -m pytest datasetforge/tests/ -q
```

Очікувано: `67 passed`. Якщо червоне — далі не йти.

Заодно можна побачити, які комбінації камери виживають pixel-budget (це те, що
робить `render_runner` перед рендером):

```bash
python - <<'PY'
import yaml
from datasetforge.engine.camera_sampler import build_grid_from_config, filter_viable, estimate_target_px
cfg = yaml.safe_load(open('datasetforge/configs/v1_light_vehicle.yaml'))
grid = build_grid_from_config(cfg['camera'])
img_w = cfg['image_size'][0]; tgt = cfg['class']['target_size_m']; ms = cfg['output']['min_side_px']
viable, rej = filter_viable(grid, img_w, tgt, ms)
print(f"viable {len(viable)}/{len(grid)} (rejected {len(rej)} як занадто дрібні)")
for s in viable:
    lo, mi = estimate_target_px(s, img_w, tgt)
    print(f"  alt={s.altitude_m:.0f}m ang={s.view_angle_deg:.0f}° hfov={s.hfov_deg:.0f}° d={s.distance_m:.0f}m → {lo:.0f}x{mi:.0f}px")
PY
```

Якщо тут порожньо або цілі <10px — генерувати нема сенсу, спочатку крути камеру.

---

## 1. Blender-only smoke (головний тест, без RunPod)

### 1.1 Оточення

Потрібні `blenderproc`, `opencv-python-headless`, `pillow`, `pyyaml`, `numpy`.
(`diffusion.enabled: false` за замовчуванням — Flux/torch НЕ потрібні для Stage 1.)

```bash
pip install blenderproc opencv-python-headless pillow pyyaml numpy matplotlib
```

> BlenderProc сам стягне Blender (~330 MB) при першому `blenderproc run`. На CPU
> кадр 1920×1080 рендериться повільно (десятки секунд–хвилини) — для smoke беремо
> `--n 8..16`. Оклюжн подвоює час (two-pass), тому для першого прогону можна
> тимчасово вимкнути `occlusion.enabled: false`.

### 1.2 Розкладка ассетів

`--assets-root` має містити:

```
<assets>/
  models/<class_name>/         *.glb|.gltf|.fbx|.obj|.blend   (напр. light_vehicle/)
  hdri/<season>/               *.hdr|*.exr                    (summer, autumn_mud, winter, spring)
  textures/ground/<season>/    *_diff_*.png|*_diff_*.jpg
```

`<class_name>` = `class.name` з конфіга (`light_vehicle`, `ifv_apc`).
`<season>` — усі значення зі `scene.seasons`.

### 1.3 Прогони (обери варіанти під те, що перевіряєш)

**A. Базовий (aabb, оклюжн вимкнено) — перевірка масштабу цілей:**

```bash
blenderproc run datasetforge/engine/render_runner.py \
  --config datasetforge/configs/v1_light_vehicle.yaml \
  --n 12 --out out/smoke_base \
  --assets-root datasetforge/assets --seed 42
```

**B. З оклюжном (дерева/кущі/сітки + visibility) — головна нова фіча:**

Переконайся, що в конфізі `occlusion.enabled: true` (дефолт), і запусти в нову папку:

```bash
blenderproc run datasetforge/engine/render_runner.py \
  --config datasetforge/configs/v1_light_vehicle.yaml \
  --n 16 --out out/smoke_occ \
  --assets-root datasetforge/assets --seed 100
```

У логах шукай рядки `[occluder] built N meshes`, `[discard] ... visibility X < 0.25`,
і підпис `occ=k vis=0.xx` у `[k/n]`.

**C. Wreck + hard-negative — перевірка «розбита ≠ ціла»:**

Використай APC-конфіг (там `destroyed.enabled: true, ratio 0.15` і `hard_negatives 0.15`):

```bash
blenderproc run datasetforge/engine/render_runner.py \
  --config datasetforge/configs/v1_apc_reference.yaml \
  --n 20 --out out/smoke_wreck \
  --assets-root datasetforge/assets --seed 7
```

Шукай у логах `[wreck] mode=hn charred+toppled`, `HARD-NEG`, `WRECK:hn`.

**D. OBB замість AABB — орієнтований bbox для танк/БТР:**

```bash
# разова копія конфіга з obb (щоб не чіпати основний):
python - <<'PY'
import yaml
c = yaml.safe_load(open('datasetforge/configs/v1_apc_reference.yaml'))
c['output']['bbox_format'] = 'obb'
yaml.safe_dump(c, open('out/apc_obb.yaml','w'))
PY
mkdir -p out
blenderproc run datasetforge/engine/render_runner.py \
  --config out/apc_obb.yaml \
  --n 12 --out out/smoke_obb \
  --assets-root datasetforge/assets --seed 5
```

Labels стануть 9-колонковими (`cls x1 y1 x2 y2 x3 y3 x4 y4`).

---

## 2. Візуальна верифікація

Скрипт `datasetforge/tools/preview_grid.py` будує контактку і зведену статистику
з будь-якої рендер-папки. Bbox-формат (aabb/obb), оклюжн, wreck, HN — читаються з
metadata автоматично.

```bash
# контактка (усі кадри в один PNG) + текстова зведена:
python datasetforge/tools/preview_grid.py --out out/smoke_occ

# ще й детальний 4-up на кожен кадр (RGB+bbox / mask / depth):
python datasetforge/tools/preview_grid.py --out out/smoke_occ --per-frame

# лише текстова зведена (швидко, без картинок):
python datasetforge/tools/preview_grid.py --out out/smoke_occ --no-sheet
```

Виходи:
- `out/<run>/_sheet.png` — контактка з підписами (alt/angle/hfov, est px, `occ/vis`, `WRECK`, `HARD-NEG`);
- `out/<run>/_preview/<stem>.png` — по-кадрові 4-up (з `--per-frame`);
- текстова зведена у stdout — розподіл HN/wreck/occluders, min-visibility, min-side px.

Приклад зведеної:
```
=== out/smoke_occ — 16 frames ===
  hard_neg      : 2 (12%)
  wreck         : 0 (0%)
  with occluders: 8 (50%)
  empty label   : 2 (має дорівнювати hard_neg + wreck-hn)
  visibility    : min=0.31 mean=0.68 (усі мають бути ≥ min_visible_frac)
  est min-side  : min=6.1px max=14.1px (усі ≥ min_side_px, інакше «крапки»)
  bbox_format   : {'aabb'}
```

---

## 3. Чек-ліст приймання (дивись на `_sheet.png` + зведену)

Пройде на RunPod лише те, що зелене тут:

**Масштаб (фікс «крапок»):**
- [ ] Техніка ВИДНА як силует (корпус/колеса/ствол), не піксельна пляма.
- [ ] `est min-side` у зведеній ≥ `min_side_px` для всіх не-HN кадрів.
- [ ] bbox щільно обгортає техніку, без великого «повітря» (для oblique — розглянь OBB).

**Оклюжн (нова фіча):**
- [ ] На occ-кадрах реально видно дерево/кущ/сітку, що **перекриває частину** техніки (не «поряд у полі»).
- [ ] bbox лишається **amodal** — обгортає ВЕСЬ силует, навіть перекриту частину (не стискається до видимого шматка).
- [ ] `visibility` мінімум ≥ `min_visible_frac` (0.25); сильно перекриті кадри у логах позначені `[discard]`.
- [ ] `with occluders` ≈ `occlusion.ratio` (0.5).

**Wreck / hard-negative:**
- [ ] WRECK-кадри: техніка обгоріла (темна) і перекинута/на боці — силует «розбитий».
- [ ] `wreck_mode: hn` та HARD-NEG кадри мають **порожній** label (жодного bbox).
- [ ] `empty label` = `hard_neg` + wreck-hn (жодних «зайвих» порожніх — це б означало отруєні false-negatives).

**Ракурси/умови:**
- [ ] Різні alt/angle/hfov (не всі однакові); nadir (90°) і oblique присутні.
- [ ] Сезони/погода/ландшафт варіюються; техніка стоїть на землі (не тоне, не летить).

**OBB (якщо `bbox_format: obb`):**
- [ ] Полігон повернутий уздовж корпусу, а не осьовий прямокутник.

Якщо щось червоне — правимо конфіг/код і **повторюємо Stage 1**, не витрачаючи RunPod.

---

## 4. (Опційно) Diffusion smoke — 1 кадр з Flux

Коли Blender-вихід чистий, перевір фотореалізм фону на ОДНОМУ кадрі, перш ніж
ганяти сотні. Це вже потребує GPU (Kaggle P100/16GB або RunPod).

- **Kaggle**: `datasetforge/pipelines/colab/blender_flux_kaggle_v2.ipynb` —
  Cell 5 рендерить 1 кадр (вмикає `diffusion.enabled`), Cell 6 показує RGB+bbox/
  depth/normals/mask, Cell 8-9 роблять inpaint + composite з 3-up прев'ю і
  pixel-identity перевіркою техніки.
- Дивись: фон фотореалістичний (не «пластик»), техніка НЕ перемальована Flux
  (pixel-identity ок), depth не саттурить (не суцільний жовтий).

> Примітка: `inference_size` тепер `768×1344` (16:9 під сенсор), а не квадрат —
> кадр не сквошиться. Якщо міняєш роздільність — став кратну 16.

---

## 5. RunPod повний ран

Лише після зеленого чек-ліста (і, бажано, чистого diffusion-smoke).

```bash
# на поді:
export REPO_DIR=/workspace/yolo-bluebierd HF_TOKEN=hf_...
bash $REPO_DIR/datasetforge/pipelines/runpod/setup.sh   # deps + прогрів ваг Flux
# далі — notebook_fluxfill.ipynb / notebook_qwen2509.ipynb / notebook.ipynb
```

Перед масштабним прогоном на поді ПОВТОРИ Stage 1+2 (`--n 16`, `preview_grid.py`)
вже на подовому залізі — швидко, а ловить проблеми з ассетами/шляхами до того,
як запустиш повну генерацію на сотні кадрів.

Після повного рану проганяй `preview_grid.py --no-sheet` по фінальній папці —
зведена одразу покаже, чи розподіл HN/wreck/occluders/visibility такий, як у конфізі.

---

## Довідка: що впливає на що (конфіг → вихід)

| Конфіг-ключ | Ефект |
|---|---|
| `image_size` | роздільність рендера (1920×1080 = реальний сенсор; менше = «крапки») |
| `class.target_size_m` | реальна довжина техніки, нормалізація моделі (Tigr 5.7, БТР 7.7) |
| `camera.altitude_m / view_angle_deg / hfov_deg` | грід ракурсів; нежиттєздатні комбінації відсіюються pixel-budget |
| `output.min_side_px` | поріг bbox у px рендера; менше — discard кадру |
| `output.bbox_format` | `aabb` \| `obb` |
| `occlusion.*` | дерева/кущі/сітки + two-pass visibility + поріг `min_visible_frac` |
| `destroyed.*` | wreck: `mode: hn` (без боксу) \| `class` (свій `class_id`) |
| `hard_negatives.ratio` | частка порожніх кадрів без техніки |
| `polish.*` | камерна деградація (Stage 5, albumentations) |

Деталі піксельного бюджету і порогів — `docs/pixel_budget.md`;
правила розмітки (amodal, occlusion 25%, BDA-ознаки, танк vs БТР) —
`docs/labeling_guidelines.md`.
```
