# datasetforge roadmap (HF diffusion approach)

## Стек

Усе через **HuggingFace** + **HF Jobs**. Без Blender, без купівлі 3D-моделей.

```
HF model       Role
─────────────────────────────────────────────────────
FLUX.1-dev     base image generation (better than SDXL для realism)
SDXL           fallback / швидше для batch
LoRA (own)     fine-tune на synthetic_apc_726 style
GLIGEN         text + bbox conditional (тре генерує обʼєкт у конкретному bbox)
GroundingDINO  open-vocab detection — для bbox refinement
SAM2           segmentation refinement (опційно)
```

## Phase 0 — HF Foundation (1-2 дні)

- HF token у `.env` (вже є).
- `hf` CLI + skill `huggingface-skills:hugging-face-jobs` готові.
- Перший test: `FLUX.1-dev` згенерує одне зображення з prompt `"Russian T-72 tank in muddy Ukrainian field, aerial drone view, 400m altitude, oblique angle"` — sanity.
- Запустити через `hf jobs run` на A10G/T4.

## Phase 1 — GLIGEN bbox-conditional MVP (3-5 днів)

- GLIGEN pipeline через `diffusers`:
  ```python
  pipe = StableDiffusionGLIGENPipeline.from_pretrained(
      "masterful/gligen-1-4-generation-text-box", ...
  )
  images = pipe(
      prompt="Russian T-72 tank in muddy field, aerial view, 400m, oblique",
      gligen_phrases=["T-72 tank"],
      gligen_boxes=[[0.3, 0.4, 0.55, 0.6]],   # YOLO bbox у normalized (xmin, ymin, xmax, ymax)
      num_images_per_prompt=4,
  ).images
  ```
- Output: image + ВЖЕ ВІДОМИЙ bbox (не треба annotation step).
- Sanity 100 кадрів tank, рандом altitude/angle/landscape.
- Інспекція через `dataset/inspect.py` — distribution має бути similar до `synthetic_apc_726`.

## Phase 2 — Style LoRA на synthetic_apc_726 (1 тиж)

- Fine-tune **SDXL LoRA** на 726 кадрах APC.
  - `huggingface-skills:hugging-face-model-trainer` skill — підтримує LoRA.
  - HF Jobs з A100, ~1-2 години, $5-10.
- LoRA вивчає: oblique drone perspective, GSD distribution, lighting, текстури.
- Result: prompts тепер генерують зображення в правильному стилі.

## Phase 3 — Batch для 10 класів (1-2 тиж)

- `prompts/<class>.yaml` per клас з вariantами скриптів:
  ```yaml
  class: tank
  models: [T-72B3, T-80, T-90]
  prompts:
    - "Russian {model} on green Ukrainian wheat field, summer, sunny, aerial drone view {altitude}m, oblique {angle} degrees"
    - "Russian {model} in muddy autumn field, rasputitsa, overcast, drone view {altitude}m, oblique {angle}"
    - "Russian {model} on snow-covered ground with shelterbelt nearby, winter, drone view {altitude}m, oblique {angle}"
  variations:
    altitude: [300, 400, 600, 800]
    angle: [15, 30, 45, 60]
  ```
- Orchestrator: `pipelines/generate_class.py` — батч 1200 кадрів per клас.
- HF Job parallel per клас.

## Phase 4 — Refinement через GroundingDINO + degradation (паралельно)

- `auto_annotation/refine.py` — пропускає згенеровані кадри через GroundingDINO для verification:
  - Якщо bbox confidence низька → відкидаємо кадр.
  - Якщо знайдено additional obj of interest → додаємо bbox.
- `degradation/`:
  - `motion_blur.py` — kernel 3-9 px з рандом direction
  - `jpeg.py` — q=60-95 рандом
  - `atmosphere.py` — fog/haze overlay через PIL alpha blend

## Phase 5 — Hard negatives (паралельно)

- Окремі промпти без військ техніки:
  - "Empty Ukrainian wheat field, aerial drone view 400m"
  - "Forest belt edge, drone view, winter, snow"
  - "Destroyed burned tank wreckage, aerial drone view, smoke" (без bbox — це shum)
  - "Civilian truck on road, aerial drone view"
- ~2000 кадрів = 15-20% датасета.

## Compute & cost

| Item | Est. |
|---|---|
| Phase 0 sanity | $1-2 |
| Phase 1 MVP 100 frames | $5 |
| Phase 2 LoRA fine-tune | $5-10 |
| Phase 3 batch 12k frames | $30-50 |
| Phase 4 GroundingDINO refine 12k | $5-10 |
| **Total** | **~$50-100** |

Все на HF Jobs. Без локального GPU.

## Timeline

**3-4 тижні до v1.0.0** на одного programmistа.

## Що паралельно

- `training/` skeleton (yolo11 train loop, HF Jobs orchestration)
- `evaluation/` skeleton (mAP, per-class, real vs synthetic split)
- `inference/` + `aim_assist/` skeletons
- OSINT eval collection (~200-500 real drone frames)

## Що НЕ блокує

- Brave1 timeline — паралельний трек
- Public datasets ~68k — pretrain backbone тільки, опційно
