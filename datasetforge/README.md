# datasetforge

Синтетичний генератор кадрів через **відкриті diffusion-моделі з HuggingFace**, з автоматичною bbox-генерацією.

> Заміняю Blender-пайплайн який був надмірно складним. Тут — diffusion + LoRA + auto-annotation.

## Стек

| Шар | Інструмент | HF model |
|---|---|---|
| **Base generation** | FLUX.1-dev або SDXL | `black-forest-labs/FLUX.1-dev`, `stabilityai/stable-diffusion-xl-base-1.0` |
| **Style learning** | LoRA fine-tune на `synthetic_apc_726` | own LoRA |
| **Direct bbox control** | InstanceDiffusion / GLIGEN | `gligen/diffusers-generation-text-box` |
| **Auto-annotation** | GroundingDINO + SAM2 | `IDEA-Research/grounding-dino-tiny`, `facebook/sam2-hiera-large` |
| **Compute** | HF Jobs з GPU | T4/A10G/A100 |

## Чому це працює

1. **Bbox умова при генерації** (GLIGEN/InstanceDiffusion): даєш prompt + bbox координати → отримуєш image з обʼєктом саме там. Bbox label = автомат.
2. **GroundingDINO для fallback**: якщо генерація без bbox-кондиціонування — генеруємо вільно, потім ground-DINO находить bbox.
3. **LoRA fine-tune на synthetic_apc_726**: ловимо стиль референсного датасета (oblique drone, 300-800м, distribution розмірів).

## Pipeline

```
prompts.yaml (class, scene, season, ...)
      │
      ▼
┌──────────────────────┐
│ HF Job на GPU        │
│   1. SDXL/FLUX + LoRA│  base image (без військ техніки)
│   2. GLIGEN/Instance │  додати ціль на конкретний bbox
│   3. degradation     │  jpeg, blur, atmosphere
│   4. ground-DINO     │  sanity-check + bbox refinement
└──────────┬───────────┘
           ▼
      image + YOLO label + metadata.json
           ▼
   data/datasetforge_v1/{train,valid,test}/
```

## Цільові характеристики

| | Цільове |
|---|---|
| Image size | 1024×1024 (downscale 640 на тренуванні) |
| Bbox min-side px | ≥10 (детекція), ≥20 (класифікація) |
| Bbox median | ~60 px |
| Стратифікація | висота × кут × сезон × ландшафт |
| Метадані | JSON sidecar обовʼязковий |

## Папки

| | Призначення |
|---|---|
| `engine/` | Diffusion pipeline wrappers, GLIGEN integration |
| `prompts/` | YAML prompt templates per class+condition |
| `lora/` | LoRA weights (gitignored, push в HF) |
| `auto_annotation/` | GroundingDINO + SAM2 для refine |
| `degradation/` | Drone-realism postprocessing |
| `output/` | YOLO writer + metadata |
| `tests/` | Sanity tests |

## Roadmap (real)

| Phase | Зусилля | Що |
|---|---|---|
| 0 | 1-2 дні | HF Jobs setup, FLUX/SDXL baseline render на тестовому prompt |
| 1 | 3-5 днів | GLIGEN bbox-conditional generation 1 клас (tank) — sanity 100 кадрів |
| 2 | 1 тиж | LoRA fine-tune на synthetic_apc_726 → style transfer |
| 3 | 1-2 тиж | Усі 10 класів × 1200 кадрів = ~12k. HF Jobs batch. |
| 4 | iteration | GroundingDINO refine + degradation tuning |

**Total ~3-4 тижні до v1.0.0** на одного. Без купівлі 3D-моделей. Без Blender.

## Compute estimate

- FLUX-1.0-dev на A10G: ~5 сек/кадр. 12k кадрів = ~17 годин = **~$30** на HF Jobs.
- LoRA fine-tune: ~1-2 години на A100 = **~$5-10**.
- GroundingDINO refine: CPU-friendly, безкоштовно якщо локально (але у нас Phenom II — теж на HF Jobs).

**Total ~$50-100** vs $200-1000 за 3D-моделі + місяці на Blender pipeline.

## Що НЕ робимо

- ❌ Blender + 3D моделі
- ❌ Сатвлені сценіreference (TurboSquid/CGTrader)
- ❌ Photogrammetry
- ❌ Власна моделізація

## Стан

🟢 Архітектура: оновлено
🔴 HF Jobs пайплайн: TODO Phase 0
🔴 LoRA fine-tune: TODO Phase 2
🔴 GLIGEN integration: TODO Phase 1
