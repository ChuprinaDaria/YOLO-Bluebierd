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

import cv2
import numpy as np
import torch
from PIL import Image

from datasetforge.pipelines.inpaint.prompts import build_prompt


def load_pipeline(diffusion_cfg: dict, device: str = "cuda"):
    """Завантажує FluxControlInpaintPipeline у GPU. bf16, no offload."""
    from diffusers import FluxControlInpaintPipeline

    pipe = FluxControlInpaintPipeline.from_pretrained(
        diffusion_cfg["base_model"],
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return pipe


def _load_depth_normalized(depth_path: Path, target_hw: tuple[int, int]) -> Image.Image:
    """Load 16-bit depth PNG, clip per-frame 1-99 percentile, normalize [0,1], stack 3ch."""
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"depth not loaded: {depth_path}")
    depth_m = depth_raw.astype(np.float32) / 1000.0
    h, w = target_hw
    depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)
    finite = depth_resized[np.isfinite(depth_resized)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if hi - lo < 1e-6:
            hi = lo + 1.0
    depth_norm = np.clip((depth_resized - lo) / (hi - lo), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    depth_3ch = np.stack([depth_uint8] * 3, axis=-1)
    return Image.fromarray(depth_3ch)


def _build_inpaint_mask(mask_path: Path, target_hw: tuple[int, int],
                        dilate_px: int, feather_px: int) -> Image.Image:
    """Build FLUX inpaint mask: 255 over BACKGROUND (inpaint), 0 over VEHICLE (keep).

    Stage 1 mask = vehicle@255. Тут: dilate+feather vehicle area, потім INVERT —
    щоб vehicle лишився заморожений, а bg перемалювався.
    """
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"mask not loaded: {mask_path}")
    h, w = target_hw
    mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    veh_mask = (mask_resized >= 128).astype(np.uint8) * 255
    if dilate_px > 0:
        k = dilate_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        veh_mask = cv2.dilate(veh_mask, kernel, iterations=1)
    if feather_px > 0:
        sigma = float(feather_px)
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        veh_mask = cv2.GaussianBlur(veh_mask, (ksize, ksize), sigma)
    # Invert: FLUX inpaint expects 255=inpaint, 0=keep.
    inpaint_mask = 255 - veh_mask
    return Image.fromarray(inpaint_mask)


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
    # FLUX pipelines у diffusers >=0.31 приймають negative_prompt;
    # якщо version не підтримує — fallback без.
    try:
        result = pipe(negative_prompt=negative, **call_kwargs).images[0]
        used_negative = True
    except TypeError:
        result = pipe(**call_kwargs).images[0]
        used_negative = False

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
        "depth_percentile_clip": [1.0, 99.0],
    }
