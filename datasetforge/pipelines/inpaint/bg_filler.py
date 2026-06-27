"""Flux depth-conditioned background inpaint stage.

Використовує `black-forest-labs/FLUX.1-Depth-dev` як інпейнт-базу — це
офіційний BFL model з вбудованим depth conditioning (не окремий ControlNet).
Завантажується одним викликом, depth подається через `control_image`.

Vehicle pixels frozen через mask INVERSION (FLUX inpaint convention:
mask=255 → inpaint, mask=0 → keep). Stage 1 mask = vehicle@255, тому
у `_build_inpaint_mask` інвертуємо: vehicle stays, bg gets repainted.

RunPod A100/H100/Blackwell 80GB+, bf16, no offload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from datasetforge.pipelines.shared.prompts import build_prompt
from datasetforge.pipelines.shared.cond import (
    build_inpaint_mask as _build_inpaint_mask,
    load_depth_normalized as _load_depth_normalized,
)


def load_pipeline(diffusion_cfg: dict, device: str = "cuda"):
    """Завантажує FluxControlInpaintPipeline у GPU. bf16, no offload.

    Опційно вантажить realism-LoRA (`diffusion_cfg["lora"]`) — це найсильніший
    важіль проти «пластику», сильніший за формулювання промпта. LoRA треновані
    на FLUX.1-dev зазвичай застосовні і до Depth-dev (ті самі attention-шари),
    але не гарантовано — тому під try/except, фейл не валить pipeline.
    """
    from diffusers import FluxControlInpaintPipeline

    from datasetforge.pipelines.shared.precision import select_precision
    plan = select_precision(force=diffusion_cfg.get("force_precision"))
    print(f"[precision] {plan}")

    pipe = FluxControlInpaintPipeline.from_pretrained(
        diffusion_cfg["base_model"],
        torch_dtype=torch.bfloat16,
    )
    if plan["offload"]:
        pipe.enable_sequential_cpu_offload()   # Kaggle 16GB — повільно, але влазить
    else:
        pipe.to(device)

    lora_cfg = diffusion_cfg.get("lora") or {}
    if lora_cfg.get("enabled") and lora_cfg.get("repo"):
        adapter = str(lora_cfg.get("adapter_name", "realism"))
        scale = float(lora_cfg.get("scale", 0.8))
        kw = {}
        if lora_cfg.get("weight_name"):
            kw["weight_name"] = lora_cfg["weight_name"]
        try:
            pipe.load_lora_weights(lora_cfg["repo"], adapter_name=adapter, **kw)
            # set_adapters виставляє глобальну силу LoRA — не треба joint_attention scale.
            pipe.set_adapters([adapter], adapter_weights=[scale])
            print(f"[lora] loaded {lora_cfg['repo']} "
                  f"(weight={lora_cfg.get('weight_name', 'auto')}) scale={scale}")
        except Exception as exc:
            print(f"[lora] FAILED ({exc.__class__.__name__}: {exc}) — "
                  f"продовжуємо без LoRA")
    return pipe


def inpaint_one(
    pipe,
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    meta_path: Path,
    out_path: Path,
    diffusion_cfg: dict,
) -> dict[str, Any]:
    """Один кадр: RGB + depth-conditioning + frozen-vehicle-mask → AI-bg PNG.

    Vehicle area може мати artifacts на edges (feathered зона) — це OK, бо
    Stage 4 composite використовує undilated binary mask і повертає vehicle
    pixels назад інтактними.
    """
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    inf_h, inf_w = diffusion_cfg["inference_size"]

    rgb = Image.open(rgb_path).convert("RGB").resize((inf_w, inf_h), Image.LANCZOS)
    depth_ctrl = _load_depth_normalized(depth_path, (inf_h, inf_w))
    mask_inpaint = _build_inpaint_mask(
        mask_path, (inf_h, inf_w),
        diffusion_cfg["mask_dilate_px"],
        diffusion_cfg["mask_feather_px"],
    )

    positive, negative = build_prompt(metadata, diffusion_cfg)
    seed = int(metadata.get("seed", 0)) + int(diffusion_cfg["seed_offset"])
    generator = torch.Generator("cpu").manual_seed(seed)

    call_kwargs = dict(
        prompt=positive,
        image=rgb,
        mask_image=mask_inpaint,
        control_image=depth_ctrl,
        strength=float(diffusion_cfg.get("strength", 1.0)),
        guidance_scale=float(diffusion_cfg["guidance"]),
        num_inference_steps=int(diffusion_cfg["steps"]),
        height=inf_h,
        width=inf_w,
        generator=generator,
    )

    # FLUX-dev — guidance-distilled: negative_prompt діє ТІЛЬКИ коли true_cfg_scale>1
    # (інакше CFG не рахується і негатив ігнорується). true_cfg_scale>1 ≈ ×2 час.
    # Передаємо лише ті kwargs, які реально приймає __call__ цієї версії pipeline.
    import inspect
    sig_params = inspect.signature(pipe.__call__).parameters
    true_cfg = float(diffusion_cfg.get("true_cfg_scale", 1.0) or 1.0)
    used_negative = False
    if "negative_prompt" in sig_params:
        call_kwargs["negative_prompt"] = negative
        used_negative = True
    cfg_active = False
    if "true_cfg_scale" in sig_params and true_cfg > 1.0:
        call_kwargs["true_cfg_scale"] = true_cfg
        cfg_active = True

    result = pipe(**call_kwargs).images[0]
    # negative «ефективний» лише якщо і переданий, і CFG увімкнено.
    negative_effective = used_negative and cfg_active

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    return {
        "base_model": diffusion_cfg["base_model"],
        "pipeline": diffusion_cfg["pipeline"],
        "inference_size": [inf_h, inf_w],
        "steps": int(diffusion_cfg["steps"]),
        "guidance": float(diffusion_cfg["guidance"]),
        "strength": float(diffusion_cfg.get("strength", 1.0)),
        "mask_dilate_px": int(diffusion_cfg["mask_dilate_px"]),
        "mask_feather_px": int(diffusion_cfg["mask_feather_px"]),
        "seed": seed,
        "prompt": positive,
        "negative_prompt": negative if used_negative else None,
        "true_cfg_scale": true_cfg,
        "negative_effective": negative_effective,
        "lora": (diffusion_cfg.get("lora") or {}).get("repo")
                if (diffusion_cfg.get("lora") or {}).get("enabled") else None,
        "depth_percentile_clip": [1.0, 99.0],
    }
