"""Pipeline #3 — FLUX.1-Fill-dev mask-inpaint background.

`FluxFillPipeline` (diffusers >=0.32). Чистий mask-inpaint: image + mask_image
(255=inpaint фон, 0=keep техніка) + prompt. БЕЗ depth/control (на відміну від
FLUX-Depth #1). Підтримує negative_prompt + true_cfg_scale і realism-LoRA.

⚠️ Ліцензія FLUX.1-dev — NON-COMMERCIAL. ОК для тесту якості; для BlueBird-продакшну
потрібна комерційна ліцензія BFL. (Qwen #2 — Apache-2.0, чистий.)

VRAM-aware через shared.precision (Kaggle 16GB ↔ RunPod 80GB).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from datasetforge.pipelines.shared.prompts import build_prompt
from datasetforge.pipelines.shared.cond import build_inpaint_mask
from datasetforge.pipelines.shared.precision import select_precision


def load_pipeline(diffusion_cfg: dict, device: str = "cuda"):
    """Завантажує FluxFillPipeline з авто-precision + опційним realism-LoRA."""
    from diffusers import FluxFillPipeline

    plan = select_precision(force=diffusion_cfg.get("force_precision"))
    print(f"[precision] {plan}")
    base = diffusion_cfg["base_model"]

    transformer = None
    if plan["gguf"] and (diffusion_cfg.get("gguf") or {}).get("url"):
        try:
            from diffusers import GGUFQuantizationConfig, FluxTransformer2DModel
            transformer = FluxTransformer2DModel.from_single_file(
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
    pipe = FluxFillPipeline.from_pretrained(base, **kw)

    if plan["offload"]:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    # Realism-LoRA (FLUX.1-dev LoRA сумісні з Fill — спільна архітектура). Guarded.
    lora_cfg = diffusion_cfg.get("lora") or {}
    if lora_cfg.get("enabled") and lora_cfg.get("repo"):
        adapter = str(lora_cfg.get("adapter_name", "realism"))
        scale = float(lora_cfg.get("scale", 0.8))
        kw_l = {"weight_name": lora_cfg["weight_name"]} if lora_cfg.get("weight_name") else {}
        try:
            pipe.load_lora_weights(lora_cfg["repo"], adapter_name=adapter, **kw_l)
            pipe.set_adapters([adapter], adapter_weights=[scale])
            print(f"[lora] loaded {lora_cfg['repo']} scale={scale}")
        except Exception as exc:
            print(f"[lora] FAILED ({exc.__class__.__name__}: {exc}) — без LoRA")
    return pipe


def fill_one(
    pipe,
    rgb_path: Path,
    depth_path: Path,       # не використовується (Fill без depth); для parity у notebook
    mask_path: Path,
    meta_path: Path,
    out_path: Path,
    diffusion_cfg: dict,
) -> dict[str, Any]:
    """Один кадр: RGB + inverted-mask inpaint → ai_bg. Depth ігнорується."""
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    inf_h, inf_w = diffusion_cfg["inference_size"]

    rgb = Image.open(rgb_path).convert("RGB").resize((inf_w, inf_h), Image.LANCZOS)
    mask_inpaint = build_inpaint_mask(
        mask_path, (inf_h, inf_w),
        int(diffusion_cfg.get("mask_dilate_px", 4)),
        int(diffusion_cfg.get("mask_feather_px", 2)),
    )
    positive, negative = build_prompt(metadata, diffusion_cfg, mode="describe")
    seed = int(metadata.get("seed", 0)) + int(diffusion_cfg.get("seed_offset", 3000))
    generator = torch.Generator("cpu").manual_seed(seed)

    sig = inspect.signature(pipe.__call__).parameters
    call_kwargs: dict[str, Any] = dict(
        prompt=positive,
        image=rgb,
        mask_image=mask_inpaint,
        guidance_scale=float(diffusion_cfg["guidance"]),
        num_inference_steps=int(diffusion_cfg["steps"]),
        height=inf_h,
        width=inf_w,
        generator=generator,
    )
    if "max_sequence_length" in sig:
        call_kwargs["max_sequence_length"] = int(diffusion_cfg.get("max_sequence_length", 512))

    true_cfg = float(diffusion_cfg.get("true_cfg_scale", 1.0) or 1.0)
    used_negative = cfg_active = False
    if "negative_prompt" in sig:
        call_kwargs["negative_prompt"] = negative
        used_negative = True
    if "true_cfg_scale" in sig and true_cfg > 1.0:
        call_kwargs["true_cfg_scale"] = true_cfg
        cfg_active = True

    result = pipe(**call_kwargs).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    return {
        "base_model": diffusion_cfg["base_model"],
        "pipeline": diffusion_cfg.get("pipeline", "FluxFillPipeline"),
        "inference_size": [inf_h, inf_w],
        "steps": int(diffusion_cfg["steps"]),
        "guidance": float(diffusion_cfg["guidance"]),
        "true_cfg_scale": true_cfg,
        "negative_effective": used_negative and cfg_active,
        "lora": (diffusion_cfg.get("lora") or {}).get("repo")
                if (diffusion_cfg.get("lora") or {}).get("enabled") else None,
        "seed": seed,
        "prompt": positive,
        "mode": "describe",
    }
