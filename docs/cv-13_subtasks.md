# CV-13 — Subtasks skeleton для Jira

> Локально в `docs/` (gitignored разом з `docs/epic.md`). Завести у Jira окремо.

Структуровано під 4 групи: **A. Locking**, **B. Generation**, **C. Eval+Versioning**, **D. QA**.

## Group A — Locking decisions

### CV-13.1 — Затвердити фінальну таксономію (v3)
- **Опис:** v3 пропонує 10 класів (Tier 1: tank, ifv_apc, artillery, air_defense, mlrs, truck_logistics, radar_ew; Tier 2: light_vehicle, motorcycle, infantry). Спалена техніка = hard negative.
- **Acceptance:** v3 затверджена командою (DRI: Даша + військовий експерт), занесена у Confluence/Notion.
- **Estimate:** 1 день.
- **Deps:** —
- **Refs:** `docs/taxonomy_v3.md`

### CV-13.2 — Затвердити гайдлайн розмітки v2
- **Опис:** Pixel thresholds (≥10 px detection, ≥20 px class), обовʼязкові метадані (altitude, angle, season, landscape, camera), OBB як V2-опція, hard negative bank rules.
- **Acceptance:** Підтверджено DRI. Інтегровано у workflow datasetforge.
- **Estimate:** 0.5 дня.
- **Deps:** CV-13.1.
- **Refs:** `docs/labeling_guidelines.md`

## Group B — Generation (datasetforge)

### CV-13.3 — datasetforge Phase 0: FLUX/SDXL sanity на HF Jobs
- **Опис:** Поставити HF Jobs pipeline, запустити FLUX.1-dev на test prompt `"Russian T-72 tank in muddy Ukrainian field, oblique drone 400m"`, зберегти результат.
- **Acceptance:** один кадр в `data/datasetforge_v0/test_flux.jpg` що візуально розпізнається.
- **Estimate:** 1-2 дні. Бюджет: ~$2.
- **Deps:** CV-13.1.
- **Refs:** `docs/datasetforge_roadmap.md`

### CV-13.4 — datasetforge Phase 1: GLIGEN bbox-conditional MVP
- **Опис:** Pipeline `prompt + bbox → image + label`. 100 кадрів для класу `tank`. Перевірити distribution розмірів проти `_synthetic_apc_726`.
- **Acceptance:** 100 кадрів tank, медіана bbox 40-100 px, мін ≥13 px. `dataset/inspect.py` pass.
- **Estimate:** 3-5 днів. Бюджет: ~$5.
- **Deps:** CV-13.3.

### CV-13.5 — datasetforge Phase 2: LoRA fine-tune на `_synthetic_apc_726`
- **Опис:** Fine-tune SDXL LoRA на 726 еталонних кадрах щоб перейняти стиль (oblique perspective, GSD, lighting). Через `huggingface-skills:hugging-face-model-trainer`.
- **Acceptance:** LoRA пушнута в `Dariachup/yolo-bluebierd-lora-v1` (private). Згенеровані кадри з LoRA візуально матчать референсний стиль.
- **Estimate:** 1 тиждень. Бюджет: ~$10.
- **Deps:** CV-13.4.

### CV-13.6 — datasetforge Phase 3: Batch 10 класів × 1200 кадрів
- **Опис:** Per-class YAML prompts. Orchestrator `pipelines/generate_class.py`. Стратифікація: altitude × angle × season × landscape. Hard negatives 15-20%.
- **Acceptance:** 12,000 кадрів total. Per-class баланс матчить epic priorities (більше AD/artillery/MLRS/truck/infantry). `dataset/inspect.py` усі сплити pass.
- **Estimate:** 1-2 тижні. Бюджет: ~$30-50.
- **Deps:** CV-13.5.

### CV-13.7 — GroundingDINO refinement
- **Опис:** Пропустити всі 12k кадрів через GroundingDINO для bbox sanity. Відкинути кадри з низькою confidence. Додати bbox для випадково додаткових цілей якщо знайдено.
- **Acceptance:** Drop rate <15%. Final dataset 10-11k кадрів.
- **Estimate:** Паралельно з Phase 3. Бюджет: ~$10.
- **Deps:** CV-13.6.

### CV-13.8 — Degradation pipeline
- **Опис:** Motion blur, JPEG compression artifacts, atmosphere overlays (fog/haze) — match drone-realism. Випадкові параметри per image.
- **Acceptance:** Усі кадри пройшли degradation. Visually схожі на real drone footage.
- **Estimate:** Паралельно з Phase 3. Бюджет: ~$5.
- **Deps:** CV-13.6.

## Group C — Eval + Versioning

### CV-13.9 — OSINT real-drone eval set (200-500 кадрів)
- **Опис:** Зібрати реальні drone кадри з Telegram OSINT (Madyar, Achilles, Birds of Magyar, etc.) **ТІЛЬКИ** для eval/test. Не для train. Розмітити вручну за нашою таксономією. Legally review.
- **Acceptance:** 200-500 кадрів в `data/eval_real/` з YOLO labels + metadata sidecar.
- **Estimate:** 2-3 тижні (трудомістко через legal + manual labeling).
- **Deps:** CV-13.1.

### CV-13.10 — Train/val/test split з seed
- **Опис:** Stratified split за class × altitude × season. Seed locked. `_meta/splits/` записаний.
- **Acceptance:** Кожен split покриває усі 10 класів + усі висоти + усі сезони. Спіл лок (replay-able).
- **Estimate:** 1 день.
- **Deps:** CV-13.6.

### CV-13.11 — Версіонування + пуш в HF private
- **Опис:** Пуш до `Dariachup/yolo-bluebierd-data` (приватний). Git tag `df-v1.0.0`. README з повним описом.
- **Acceptance:** Repo приватний, команда має read access. README пояснює структуру + license + статистику.
- **Estimate:** 1 день.
- **Deps:** CV-13.10.

## Group D — QA

### CV-13.12 — QA pass: per-class balance + missing labels + bbox sanity
- **Опис:** `dataset/inspect.py` + повторна валідація через скрипт. Перевірка orphan labels, дублікатів, anomaly bboxes.
- **Acceptance:** Звіт у `data/_meta/qa_v1.md`. Усі issues класифіковані як accept/fix/reject. Дrop або fix issues.
- **Estimate:** 2-3 дні.
- **Deps:** CV-13.11.

### CV-13.13 — Double-pass перевірка розмітки на sample
- **Опис:** Sample 5% кадрів. Другий аннотатор перевіряє bbox tight fit, truncation rules, class assignment. IoU consistency ≥0.85.
- **Acceptance:** Sample IoU ≥0.85. Disagreement <5%.
- **Estimate:** 1 тиждень (залежить від аннотатора).
- **Deps:** CV-13.11.

## Summary

| Group | Tickets | Total estimate |
|---|---:|---|
| A. Locking | 2 | 1.5 дня |
| B. Generation | 6 | 3-4 тижні |
| C. Eval + Versioning | 3 | 3-4 тижні (паралельно з B) |
| D. QA | 2 | 1-2 тижні |
| **Total wall clock** | **13** | **5-7 тижнів** |

Бюджет HF Jobs: **~$60-100** total.
