# YOLO-Bluebierd

Кастомна YOLO модель для розпізнавання та донаведення цілей.

## Структура

| Папка | Призначення |
|---|---|
| `dataset/` | Аудит, split, конвертація, augmentation, dataset cards |
| `training/` | Тренувальний пайплайн, конфіги, логування |
| `inference/` | Інференс, export (ONNX/TensorRT), бенчі швидкості |
| `aim_assist/` | Tracking + центрування цілі + smoothing |
| `evaluation/` | Метрики, real vs synthetic split, edge-case suite |
| `configs/` | YAML конфіги моделі, тренування, інференсу |
| `scripts/` | One-off утиліти |
| `tests/` | Юніт- та інтеграційні тести |

Кожна папка — самодостатня з власним README.

## Принципи

1. **Ідеально** — без хаків і shortcut-ів.
2. **Чисто** — чіткі межі модулів, читабельний код.
3. **Стабільно** — reproducibility (seed, версіонування датасету і ваг) закладено з першого комміту.
