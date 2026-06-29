"""Pipeline #2 — Qwen-Image-Edit-2509 (Apache-2.0) instruction-based bg edit.

`QwenImageEditPlusPipeline` (diffusers >=0.36). На відміну від FLUX-inpaint, у Qwen
НЕМА `mask_image` — це instruction-edit цілого кадру. Техніку все одно заморожує
Stage 4 composite (вставляє оригінальні пікселі по бінарній масці), тож відсутність
маски не проблема. Інструкція («лиши техніку, перероби лише фон») допомагає моделі
не перемальовувати об'єкт.

VRAM-aware (Kaggle 16GB ↔ RunPod 80GB) через shared.precision:
  bf16          → повністю на GPU                       (RunPod A100/H100)
  *_offload     → enable_sequential_cpu_offload()        (Kaggle/L4 — повільно, але влазить)
  gguf          → опційний gguf-квант трансформера, якщо заданий gguf.url

Depth-ControlNet: diffusers-API для Qwen depth ще «пливе» по версіях, тому depth
подаємо як `control_image` ЛИШЕ якщо `__call__` його приймає (інтроспекція сигнатури);
інакше тихо лишаємось на instruction-only. Прапор: diffusion_cfg["depth_control"].
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from datasetforge.pipelines.shared.prompts import build_prompt
from datasetforge.pipelines.shared.cond import load_depth_normalized
from datasetforge.pipelines.shared.precision import select_precision


def load_pipeline(diffusion_cfg: dict, device: str = "cuda"):
    """Завантажує QwenImageEditPlusPipeline з авто-вибором precision за VRAM."""
    from diffusers import QwenImageEditPlusPipeline

    plan = select_precision(force=diffusion_cfg.get("force_precision"))
    print(f"[precision] {plan}")
    base = diffusion_cfg["base_model"]

    transformer = None
    if plan["gguf"] and (diffusion_cfg.get("gguf") or {}).get("url"):
        # Опційний gguf-квант трансформера для 16GB. Best-effort — фейл → штатний шлях.
        try:
            from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
            transformer = QwenImageTransformer2DModel.from_single_file(
                diffusion_cfg["gguf"]["url"],
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            print(f"[gguf] transformer loaded: {diffusion_cfg['gguf']['url']}")
        except Exception as exc:
            print(f"[gguf] FAILED ({exc.__class__.__name__}: {exc}) — штатний bf16+offload")
            transformer = None

    kw = {"torch_dtype": torch.bfloat16}
    if transformer is not None:
        kw["transformer"] = transformer
    pipe = QwenImageEditPlusPipeline.from_pretrained(base, **kw)

    if plan["offload"]:
        pipe.enable_sequential_cpu_offload()   # peak VRAM ~ найбільший модуль
    else:
        pipe.to(device)
    return pipe


def edit_one(
    pipe,
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,        # не використовується моделлю; freeze робить Stage 4
    meta_path: Path,
    out_path: Path,
    diffusion_cfg: dict,
) -> dict[str, Any]:
    """Один кадр: RGB + instruction → відредагований кадр (ai_bg). Mask ігнорується."""
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    inf_h, inf_w = diffusion_cfg["inference_size"]

    rgb = Image.open(rgb_path).convert("RGB").resize((inf_w, inf_h), Image.LANCZOS)
    positive, negative = build_prompt(metadata, diffusion_cfg, mode="instruct")
    seed = int(metadata.get("seed", 0)) + int(diffusion_cfg.get("seed_offset", 2000))
    generator = torch.Generator("cpu").manual_seed(seed)

    sig = inspect.signature(pipe.__call__).parameters
    call_kwargs: dict[str, Any] = dict(
        image=rgb,
        prompt=positive,
        num_inference_steps=int(diffusion_cfg["steps"]),
        generator=generator,
    )
    if "guidance_scale" in sig:
        call_kwargs["guidance_scale"] = float(diffusion_cfg.get("guidance", 1.0))
    if "height" in sig:
        call_kwargs["height"] = inf_h
    if "width" in sig:
        call_kwargs["width"] = inf_w

    true_cfg = float(diffusion_cfg.get("true_cfg_scale", 4.0) or 1.0)
    used_negative = cfg_active = False
    if "negative_prompt" in sig:
        call_kwargs["negative_prompt"] = negative
        used_negative = True
    if "true_cfg_scale" in sig and true_cfg > 1.0:
        call_kwargs["true_cfg_scale"] = true_cfg
        cfg_active = True

    # Опційний depth-control — лише якщо pipeline його приймає.
    dc = diffusion_cfg.get("depth_control") or {}
    depth_used = False
    if dc.get("enabled") and "control_image" in sig and Path(depth_path).exists():
        call_kwargs["control_image"] = load_depth_normalized(depth_path, (inf_h, inf_w))
        if "controlnet_conditioning_scale" in sig:
            call_kwargs["controlnet_conditioning_scale"] = float(dc.get("scale", 0.9))
        depth_used = True

    result = pipe(**call_kwargs).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    sidecar = {
        "base_model": diffusion_cfg["base_model"],
        "pipeline": diffusion_cfg.get("pipeline", "QwenImageEditPlusPipeline"),
        "inference_size": [inf_h, inf_w],
        "steps": int(diffusion_cfg["steps"]),
        "guidance": float(diffusion_cfg.get("guidance", 1.0)),
        "true_cfg_scale": true_cfg,
        "negative_effective": used_negative and cfg_active,
        "depth_control_used": depth_used,
        "seed": seed,
        "prompt": positive,
        "mode": "instruct",
    }

    # Перезаписати metadata sidecar: render_runner Stage 1 ставить
    # diffusion.enabled=False (бо не знає чи буде Qwen). Тут чесно фіксуємо
    # actual параметри, щоб JSON відповідав реальному кадру.
    metadata_updated = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata_updated["diffusion"] = {"enabled": True, **sidecar}
    # relight + composite params з diffusion_cfg якщо є
    relight_cfg = diffusion_cfg.get("relight") or {}
    if relight_cfg:
        metadata_updated["diffusion"]["relight"] = {
            "enabled": bool(relight_cfg.get("enabled", False)),
            "strength": float(relight_cfg.get("strength", 0.0)),
            "match_color": bool(relight_cfg.get("match_color", False)),
        }
    for k in ("strength", "mask_dilate_px", "mask_feather_px"):
        if k in diffusion_cfg:
            metadata_updated["diffusion"][k] = diffusion_cfg[k]
    meta_path.write_text(
        json.dumps(metadata_updated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return sidecar
