"""Stage 4 (shared) — зшити AI-фон з технікою + узгодити світло.

Дві речі (обидві gated через diffusion_cfg):

  1. relight/harmonize — техніка рендериться Blender-світлом, фон малює модель зі
     своєю експозицією. Без узгодження «транспорт іншої яскравості ніж фон».
     `relight.match_color: true` → per-channel white-balance + exposure match.

  2. composite — alpha-blend frozen-vehicle над AI-фоном по pixel-precise масці.

Камерна «деградація» (туман/зерно/шум/jpeg) ПЕРЕНЕСЕНА у Stage 5 `polish.py`
(albumentations). Тут її більше нема — composite лишає чисту, не-зіпсовану основу.

Контракт pixel-identity:
    relight OFF → composite[mask] == rgb[mask] (byte-identical, hard assert)
    relight ON  → техніка СВІДОМО змінюється (фікс «різної яскравості»); byte-diff
                  лише логуються у stats. bbox/мітки не рухаються в обох випадках.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _luma(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sun_vec_from_meta(meta: dict) -> np.ndarray:
    """Convention: az 0° = +X (east), el = elevation above horizon (deg)."""
    az_rad = math.radians(float(meta["sun_azimuth_deg"]))
    el_rad = math.radians(float(meta["sun_elevation_deg"]))
    return np.array([
        math.cos(el_rad) * math.cos(az_rad),
        math.cos(el_rad) * math.sin(az_rad),
        math.sin(el_rad),
    ], dtype=np.float32)


def composite_one(
    rgb_path: Path,
    ai_bg_path: Path,
    mask_path: Path,
    normals_path: Path | None,
    meta_path: Path,
    out_path: Path,
    diffusion_cfg: dict,
    assert_pixel_identity: bool = True,
) -> dict[str, Any]:
    """Build composite frame. Returns stats dict для логування."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]

    ai_bg = np.array(Image.open(ai_bg_path).convert("RGB"), dtype=np.uint8)
    if ai_bg.shape[:2] != (H, W):
        ai_bg = cv2.resize(ai_bg, (W, H), interpolation=cv2.INTER_LANCZOS4)

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"mask not loaded: {mask_path}")
    if mask_raw.shape != (H, W):
        mask_raw = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bin = mask_raw >= 128  # bool — pixel-precise compose mask

    rgb_out = rgb.astype(np.float32)

    relight_cfg = diffusion_cfg.get("relight", {}) or {}
    relight_on = bool(relight_cfg.get("enabled", False))
    strength = float(relight_cfg.get("strength", 0.0))
    match_color = bool(relight_cfg.get("match_color", True))
    relight_stats: dict[str, Any] = {"enabled": relight_on, "strength": strength,
                                     "match_color": match_color}

    if relight_on and strength > 0 and mask_bin.any() and (~mask_bin).any():
        eps = 1e-6
        veh_f = rgb.astype(np.float32)
        bg_f = ai_bg.astype(np.float32)
        if match_color:
            # Per-channel white-balance + exposure: gain_c = bg_mean_c / veh_mean_c.
            veh_mean = veh_f[mask_bin].mean(axis=0) + eps        # (3,)
            bg_mean = bg_f[~mask_bin].mean(axis=0) + eps         # (3,)
            gain = np.clip(bg_mean / veh_mean, 0.5, 2.0)
            relight_stats.update({"veh_mean": veh_mean.tolist(),
                                  "bg_mean": bg_mean.tolist(),
                                  "gain_rgb": gain.tolist()})
            gain_field = gain[None, None, :]
        else:
            veh_lum = float(_luma(veh_f)[mask_bin].mean() + eps)
            bg_lum = float(_luma(bg_f)[~mask_bin].mean() + eps)
            gain = np.full(3, np.clip(bg_lum / veh_lum, 0.5, 2.0), dtype=np.float32)
            relight_stats.update({"veh_lum": veh_lum, "bg_lum": bg_lum,
                                  "exposure_global": float(gain[0])})
            gain_field = gain[None, None, :]

        modulation = np.ones((H, W), dtype=np.float32)
        if normals_path is not None and Path(normals_path).exists():
            try:
                # render_runner пише normals як 16-bit 3ch PNG, encoded (n+1)/2 * 65535.
                raw = cv2.imread(str(normals_path), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise RuntimeError("cv2.imread returned None")
                if raw.dtype == np.uint16:
                    normals = (raw.astype(np.float32) / 65535.0) * 2.0 - 1.0
                elif raw.dtype == np.uint8:
                    normals = (raw.astype(np.float32) / 255.0) * 2.0 - 1.0
                else:
                    normals = raw.astype(np.float32)
                if normals.shape[:2] != (H, W):
                    normals = cv2.resize(normals, (W, H), interpolation=cv2.INTER_LINEAR)
                if normals.ndim == 3 and normals.shape[-1] >= 3:
                    n = normals[..., :3]
                    nlen = np.linalg.norm(n, axis=-1, keepdims=True) + eps
                    n_unit = n / nlen
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    sun_vec = _sun_vec_from_meta(meta)
                    lambert = np.clip((n_unit * sun_vec).sum(axis=-1), 0.0, 1.0)
                    modulation = lambert.astype(np.float32)
                    relight_stats["modulation"] = "lambert"
            except Exception as exc:
                relight_stats["modulation"] = f"fallback-uniform ({exc.__class__.__name__})"
        else:
            relight_stats["modulation"] = "uniform"

        mod = modulation[..., None]
        scale_per_px = 1.0 + strength * (gain_field - 1.0) * mod
        rgb_out = np.clip(rgb_out * scale_per_px, 0.0, 255.0)

    mask_f = mask_bin.astype(np.float32)[..., None]
    composite = ai_bg.astype(np.float32) * (1.0 - mask_f) + rgb_out * mask_f
    composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    veh_diff = int(np.abs(rgb.astype(int) - composite_u8.astype(int))[mask_bin].max()) \
        if mask_bin.any() else 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = Image.fromarray(composite_u8)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        out_img.save(out_path, quality=92)
    else:
        out_img.save(out_path)

    # Контракт: byte-identity лише коли relight OFF (Stage 5 polish зіпсує пізніше СВІДОМО).
    harmonized = relight_on and strength > 0
    if assert_pixel_identity:
        if harmonized:
            relight_stats["veh_pixel_drift"] = veh_diff
        else:
            assert veh_diff == 0, (
                f"vehicle pixels modified (max diff={veh_diff}) when relight=OFF — "
                f"contract violation"
            )

    return {
        "image_size": [H, W],
        "vehicle_px": int(mask_bin.sum()),
        "bg_px": int((~mask_bin).sum()),
        "relight": relight_stats,
    }
