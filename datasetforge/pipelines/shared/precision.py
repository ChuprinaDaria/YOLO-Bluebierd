"""VRAM → precision selector. Kaggle (16GB) vs RunPod (80GB) один код.

Qwen-Image-Edit-2509 (20B) і FLUX.1-Fill-dev (~12B) не влазять у 16GB у bf16, тому:
  >= min_bf16_gb (40)  → bf16, повністю на GPU, без offload     (RunPod A100/H100)
  >= min_fp8_gb  (22)  → bf16 + sequential CPU offload          (24GB L4/A10)
  інакше               → gguf-Q4 + sequential CPU offload       (Kaggle T4/P100 16GB)

Чистий helper — torch імпортується lazy, щоб модуль тягнувся і без CUDA (юніт-тести).
"""

from __future__ import annotations

from typing import Any


def detect_vram_gb() -> float | None:
    """Total VRAM найбільшого видимого GPU у GB, або None якщо CUDA недоступна."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return max(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return None


def select_precision(
    vram_gb: float | None = None,
    *,
    min_bf16_gb: float = 40.0,
    min_fp8_gb: float = 22.0,
    force: str | None = None,
) -> dict[str, Any]:
    """Повертає план завантаження моделі.

    force ∈ {"bf16","fp8_offload","gguf_offload"} перевизначає авто-вибір.
    Поля:
      mode      — "bf16" | "fp8_offload" | "gguf_offload"
      dtype     — "bfloat16"
      offload   — bool (sequential CPU offload)
      gguf      — bool (вантажити gguf-quantized трансформер)
      vram_gb   — що задетектили (для логів)
    """
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    if force:
        mode = force
    elif vram_gb is None:
        # CPU-only / невідомо — найбезпечніше припустити малий бюджет.
        mode = "gguf_offload"
    elif vram_gb >= min_bf16_gb:
        mode = "bf16"
    elif vram_gb >= min_fp8_gb:
        mode = "fp8_offload"
    else:
        mode = "gguf_offload"

    return {
        "mode": mode,
        "dtype": "bfloat16",
        "offload": mode != "bf16",
        "gguf": mode == "gguf_offload",
        "vram_gb": round(vram_gb, 1) if vram_gb is not None else None,
    }
