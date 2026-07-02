"""Shared conditioning helpers — depth normalize + inverted inpaint mask.

Спільні для FLUX-Depth (#1), FLUX-Fill (#3) і Qwen depth-CN (#2).
Лише cv2/numpy/PIL — без torch, тестується локально.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_depth_normalized(depth_path: Path, target_hw: tuple[int, int]) -> Image.Image:
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


def build_inpaint_mask(mask_path: Path, target_hw: tuple[int, int],
                       dilate_px: int, feather_px: int) -> Image.Image:
    """Build inpaint mask: 255 over BACKGROUND (inpaint), 0 over VEHICLE (keep).

    Stage 1 mask = vehicle@255. Dilate+feather vehicle area, потім INVERT —
    щоб vehicle лишився заморожений, а bg перемалювався. Конвенція FLUX inpaint
    (255=inpaint, 0=keep) — спільна для FLUX-Depth і FLUX-Fill.
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
    # Invert: inpaint expects 255=inpaint, 0=keep.
    inpaint_mask = 255 - veh_mask
    return Image.fromarray(inpaint_mask)


def erase_vehicle_from_rgb(rgb_pil: Image.Image, mask_path: Path,
                            target_hw: tuple[int, int],
                            dilate_px: int = 12, radius: int = 8,
                            method: str = "TELEA") -> tuple[Image.Image, dict]:
    """Локально вирізати vehicle область з RGB (cv2.inpaint) — pre-Qwen erase.

    Без цього Qwen-Image-Edit (pure edit pipeline) бачить tank у input і клонує
    його у фон → "ghost tank" поза bbox. Заповнюємо vehicle область сусіднім
    контекстом → Qwen бачить чистий landscape → не може намалювати ще один tank.

    Dilate перед inpaint щоб ні залишити edge-hint про tank силует.

    Returns:
        (erased_rgb_pil, stats_dict_for_sidecar)
    """
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"mask not loaded: {mask_path}")
    h, w = target_hw
    mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    veh = (mask_resized >= 128).astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        veh = cv2.dilate(veh, kernel, iterations=1)
    rgb_arr = np.array(rgb_pil)
    if rgb_arr.shape[:2] != (h, w):
        rgb_arr = cv2.resize(rgb_arr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    flag = cv2.INPAINT_TELEA if method.upper() == "TELEA" else cv2.INPAINT_NS
    erased_bgr = cv2.inpaint(cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR), veh, radius, flag)
    erased_rgb = cv2.cvtColor(erased_bgr, cv2.COLOR_BGR2RGB)
    stats = {
        "enabled": True,
        "dilate_px": int(dilate_px),
        "radius": int(radius),
        "method": method.upper(),
        "mask_px": int((mask_resized >= 128).sum()),
    }
    return Image.fromarray(erased_rgb), stats
