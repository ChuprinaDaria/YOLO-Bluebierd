"""Flux depth-conditioned background inpaint stage.

Vehicle pixels frozen via dilated+feathered mask. Bbox unchanged.
Runs на RunPod A100/H100 80GB, bf16, no offload.

Public surface:
    load_pipeline(diffusion_cfg, device="cuda") -> FluxControlNetInpaintPipeline
    inpaint_one(pipe, rgb_path, depth_path, mask_path, meta_path, out_path, cfg) -> dict
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
    """Завантажує FluxControlNetInpaintPipeline + Depth ControlNet у GPU.

    Очікує що моделі вже у HF cache (setup.sh робить snapshot_download).
    bf16, full GPU (без offload — 80GB вистачає).
    """
    from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel

    cn = FluxControlNetModel.from_pretrained(
        diffusion_cfg["depth_controlnet"],
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxControlNetInpaintPipeline.from_pretrained(
        diffusion_cfg["base_model"],
        controlnet=cn,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return pipe


def _load_depth_normalized(depth_path: Path, target_hw: tuple[int, int]) -> Image.Image:
    """Load 16-bit depth PNG, clip per-frame 1-99 percentile, normalize [0,1], stack 3ch."""
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"depth not loaded: {depth_path}")
    depth_m = depth_raw.astype(np.float32) / 1000.0  # PNG зберігалось як depth_mm
    h, w = target_hw
    depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)
    # Per-frame percentile clip відсікає sky/sentinel outliers.
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
    """Bin >=128 → dilate (ellipse kernel) → Gaussian feather. Для Flux mask_image."""
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"mask not loaded: {mask_path}")
    h, w = target_hw
    mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized >= 128).astype(np.uint8) * 255
    if dilate_px > 0:
        k = dilate_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
    if feather_px > 0:
        sigma = float(feather_px)
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        mask_bin = cv2.GaussianBlur(mask_bin, (ksize, ksize), sigma)
    return Image.fromarray(mask_bin)


def inpaint_one(
    pipe,
    rgb_path: Path,
    depth_path: Path,
    mask_path: Path,
    meta_path: Path,
    out_path: Path,
    diffusion_cfg: dict,
) -> dict[str, Any]:
    """Один кадр: RGB + depth-CN + frozen-mask → AI-bg PNG.

    Vehicle pixels у output Flux MAY перемалювати — це нормально, Stage 4 (composite.py)
    бере ai_bg тільки де mask_compose=0, vehicle pixels — з raw rgb.

    Returns: sidecar dict для merge у metadata.json.
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

    result = pipe(
        prompt=positive,
        negative_prompt=negative,
        image=rgb,
        mask_image=mask_inpaint,
        control_image=depth_ctrl,
        controlnet_conditioning_scale=float(diffusion_cfg["controlnet_scale"]),
        guidance_scale=float(diffusion_cfg["guidance"]),
        num_inference_steps=int(diffusion_cfg["steps"]),
        height=inf_h,
        width=inf_w,
        generator=generator,
    ).images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    return {
        "base_model": diffusion_cfg["base_model"],
        "depth_controlnet": diffusion_cfg["depth_controlnet"],
        "pipeline": diffusion_cfg["pipeline"],
        "inference_size": [inf_h, inf_w],
        "steps": int(diffusion_cfg["steps"]),
        "guidance": float(diffusion_cfg["guidance"]),
        "controlnet_scale": float(diffusion_cfg["controlnet_scale"]),
        "mask_dilate_px": int(diffusion_cfg["mask_dilate_px"]),
        "mask_feather_px": int(diffusion_cfg["mask_feather_px"]),
        "seed": seed,
        "prompt": positive,
        "negative_prompt": negative,
        "depth_percentile_clip": [1.0, 99.0],
    }
