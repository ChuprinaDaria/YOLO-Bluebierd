# datasetforge assets — що покласти перед рендером

Ассети **gitignored** (важкі бінарники, `datasetforge/.gitignore`) — тримаємо
поза репо і кладемо вручну / стягуємо скриптом. `render_runner` шукає рівно
**3 категорії**. Оклюдери (дерева/кущі/сітки) і wreck (розбита техніка) —
**процедурні**, ассетів НЕ потребують.

## Розкладка

```
datasetforge/assets/
  models/<class_name>/         *.glb | *.gltf | *.fbx | *.obj | *.blend
  hdri/<season>/               *.hdr | *.exr
  textures/ground/<season>/    *_diff_*.png | *_diff_*.jpg      ← ім'я МУСИТЬ містити _diff_
```

`<class_name>` = `class.name` з конфіга. `<season>` = усі значення зі `scene.seasons`
(зараз: `summer`, `autumn_mud`, `winter`, `spring`).

## 1. 3D-моделі техніки — `models/<class_name>/`

`render_runner` globs **усі** файли підтримуваних форматів у теці; YAML-список
`class.models` — лише документація для людей, код його ігнорує.

| Клас | Тека | Приклади моделей |
|---|---|---|
| `light_vehicle` | `models/light_vehicle/` | GAZ Tigr, цивільні авто, mil-pickup |
| `ifv_apc` | `models/ifv_apc/` | BTR-80/82, MT-LB, BMP-1/2 |

- Мінімум для smoke: **1 модель**.
- Масштаб/одиниці не критичні — `scene_builder` нормалізує max-dim до
  `class.target_size_m` (Tigr 5.7 м, БТР 7.7 м).
- glTF Y-up конвертується у Blender Z-up автоматично.

## 2. HDRI (небо + освітлення) — `hdri/<season>/`

- Формати: `*.hdr` або `*.exr`. Мінімум **1 файл на кожен сезон**.
- Джерело: [Poly Haven HDRIs](https://polyhaven.com/hdris) — outdoor / field /
  overcast; 2K достатньо для smoke, 4K для фінального рану.

```
hdri/summer/     *.hdr
hdri/autumn_mud/ *.hdr
hdri/winter/     *.hdr
hdri/spring/     *.hdr
```

## 3. Ground-текстури — `textures/ground/<season>/`

- Формати: `*.png` / `*.jpg`, **ім'я мусить містити `_diff_`** (патерн
  `*_diff_*`) — конвенція Poly Haven (`brown_mud_diff_2k.jpg`). Файл без
  `_diff_` код **не побачить**.
- Мінімум **1 файл на сезон**.
- Джерело: [Poly Haven Textures](https://polyhaven.com/textures) —
  ground / soil / grass / snow.

```
textures/ground/summer/     *_diff_*.jpg   (трава)
textures/ground/autumn_mud/ *_diff_*.jpg   (багнюка)
textures/ground/winter/     *_diff_*.jpg   (сніг)
textures/ground/spring/     *_diff_*.jpg   (поле)
```

## Мінімальний набір для першого smoke

Обмеж один сезон у конфізі — і треба менше файлів:

```yaml
scene:
  seasons: [summer]     # тимчасово лише один
```

Тоді достатньо: **1 модель + 1 HDRI (summer) + 1 ground `*_diff_*` (summer)**,
і `--n 4` вже рендериться. Повний набір (4 сезони) = 1+ модель, 4 HDRI, 4 ground.

## Чого НЕ треба

- **Оклюдери** — процедурні примітиви (`engine/occluders.py`), без файлів.
- **Wreck / розбита техніка** — процедурний перефарб + перекид наявної моделі.
- **Road strip** — вимкнено (`scene.road_under_vehicle: false`).

## Типова помилка

Порожня тека → `render_runner` падає на `FileNotFoundError: no HDRIs in ...`
або `no ground textures in ...` (з `season_lighting.py`), або
`no .blend/.glb/... in models/<class>` — підказує, чого саме бракує.
Найчастіший недогляд — ground-текстура **без `_diff_`** у назві: файл є, а код
його не бачить.

## Скрипт-хелпер

`datasetforge/tools/fetch_assets.py` стягує безкоштовні HDRI + ground-текстури
з Poly Haven по всіх сезонах (моделі техніки — вручну, ліцензії/пошук окремо):

```bash
python datasetforge/tools/fetch_assets.py --assets-root datasetforge/assets
```
