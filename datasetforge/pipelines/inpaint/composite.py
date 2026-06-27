"""Stage 4 — зшити AI-фон з технікою + узгодити світло + «зіпсувати» під камеру.

Три речі, які робить цей крок (усі — gated через diffusion_cfg):

  1. relight/harmonize — техніка рендериться Blender-світлом, фон малює Flux зі
     своєю експозицією. Без узгодження «транспорт іншої яскравості ніж фон».
     `relight.match_color: true` → per-channel white-balance + exposure match
     (а не лише luma). Сила контролюється `relight.strength`.

  2. composite — alpha-blend frozen-vehicle над AI-фоном по pixel-precise масці.

  3. camera-realism degradation — щоб не виглядало як кінорендер, а як зйомка з
     дрона: легкий defocus/motion blur, chromatic aberration, атмосферна димка,
     vignette, сенсорний шум і JPEG-компресія. Накладається на ВЕСЬ кадр (і техніку,
     і фон) — це і зшиває їх у єдину «камеру». Gated через `diffusion_cfg["degradation"]`.

Контракт pixel-identity:
    relight OFF + degradation OFF → composite[mask] == rgb[mask] (byte-identical, hard assert)
    інакше → техніка СВІДОМО змінюється (це і є фікс «різної яскравості»); byte-diff
    лише логуються у stats, hard assert вимкнено. bbox/мітки не рухаються.

Mask береться з vehicle_masks/{stem}.png (з render_runner) і бінаризується @ 128.
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


def _apply_camera_realism(
    img_u8: np.ndarray,
    deg_cfg: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """«Зіпсувати» кадр під реальну дрон-камеру. Послідовність — оптика→сенсор→кодек.

    Усі ефекти опціональні (0/відсутність = пропустити). Геометрію bbox не змінює
    (зсуви субпіксельні / симетричні).
    """
    h, w = img_u8.shape[:2]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    img = img_u8.astype(np.float32)
    applied: dict[str, Any] = {}

    # 1. Defocus — легка м'якість фокуса (анти-«sharp focus / cinematic»).
    blur_sigma = float(deg_cfg.get("blur_sigma", 0.0) or 0.0)
    if blur_sigma > 0:
        k = max(3, int(2 * round(3 * blur_sigma) + 1))
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
        applied["blur_sigma"] = blur_sigma

    # 2. Motion blur — дрон рухається; напрямок випадковий, довжина мала.
    motion_px = int(deg_cfg.get("motion_blur_px", 0) or 0)
    if motion_px >= 2:
        angle = float(rng.uniform(0, math.pi))
        kernel = np.zeros((motion_px, motion_px), dtype=np.float32)
        cx = (motion_px - 1) / 2.0
        for t in np.linspace(-cx, cx, motion_px):
            x = int(round(cx + t * math.cos(angle)))
            y = int(round(cx + t * math.sin(angle)))
            if 0 <= x < motion_px and 0 <= y < motion_px:
                kernel[y, x] = 1.0
        s = kernel.sum()
        if s > 0:
            kernel /= s
            img = cv2.filter2D(img, -1, kernel)
            applied["motion_blur_px"] = motion_px

    # 3. Chromatic aberration — дешева оптика розводить канали на краях.
    chroma_px = float(deg_cfg.get("chroma_px", 0.0) or 0.0)
    if chroma_px > 0:
        # Масштабуємо R трохи більшим, B трохи меншим відносно центру.
        def _scale_channel(ch: np.ndarray, scale: float) -> np.ndarray:
            M = np.array([[scale, 0, (1 - scale) * w / 2.0],
                          [0, scale, (1 - scale) * h / 2.0]], dtype=np.float32)
            return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        f = chroma_px / max(w, h)
        img[..., 0] = _scale_channel(img[..., 0], 1.0 + f)   # R назовні
        img[..., 2] = _scale_channel(img[..., 2], 1.0 - f)   # B всередину
        applied["chroma_px"] = chroma_px

    # 4. Атмосферна димка — aerial haze: тягне до світло-сірого + знижує контраст.
    haze = float(deg_cfg.get("atmosphere_strength", 0.0) or 0.0)
    if haze > 0:
        haze = min(haze, 0.6)
        airlight = float(deg_cfg.get("haze_airlight", 200.0))
        img = img * (1.0 - haze) + airlight * haze
        applied["atmosphere_strength"] = haze

    # 5. Vignette — потемніння кутів (типово для wide дрон-лінз).
    vig = float(deg_cfg.get("vignette_strength", 0.0) or 0.0)
    if vig > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        r = np.clip(r / math.sqrt(2.0), 0.0, 1.0)
        mask_v = (1.0 - vig * (r ** 2))[..., None]
        img = img * mask_v
        applied["vignette_strength"] = vig

    # 6. Сенсорний шум — gaussian, переважно по яскравості + трохи хром.
    noise_sigma = float(deg_cfg.get("noise_sigma", 0.0) or 0.0)
    if noise_sigma > 0:
        lum = rng.normal(0.0, noise_sigma, size=(h, w, 1)).astype(np.float32)
        chroma = rng.normal(0.0, noise_sigma * 0.4, size=(h, w, 3)).astype(np.float32)
        img = img + lum + chroma
        applied["noise_sigma"] = noise_sigma

    out = np.clip(img, 0, 255).astype(np.uint8)

    # 7. JPEG-компресія — найсильніший «телефон/дрон» tell. Робимо останнім.
    jpeg_q = int(deg_cfg.get("jpeg_quality", 0) or 0)
    if 1 <= jpeg_q <= 100:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            applied["jpeg_quality"] = jpeg_q

    return out, applied


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
            # Узгоджує і яскравість, і колірний відтінок техніки під фон.
            veh_mean = veh_f[mask_bin].mean(axis=0) + eps        # (3,)
            bg_mean = bg_f[~mask_bin].mean(axis=0) + eps         # (3,)
            gain = bg_mean / veh_mean
            # Захист від екстремальних gain на дуже темній/насиченій техніці.
            gain = np.clip(gain, 0.5, 2.0)
            relight_stats.update({"veh_mean": veh_mean.tolist(),
                                  "bg_mean": bg_mean.tolist(),
                                  "gain_rgb": gain.tolist()})
            gain_field = gain[None, None, :]                     # broadcast (1,1,3)
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
                # cv2 round-trip bit-preserves array — channel ordering таке ж як на write.
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

        # scale = 1 + strength·(gain−1)·modulation, per-channel.
        mod = modulation[..., None]
        scale_per_px = 1.0 + strength * (gain_field - 1.0) * mod
        rgb_out = np.clip(rgb_out * scale_per_px, 0.0, 255.0)

    mask_f = mask_bin.astype(np.float32)[..., None]
    composite = ai_bg.astype(np.float32) * (1.0 - mask_f) + rgb_out * mask_f
    composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    # Драйв byte-diff техніки ДО degradation (для логів/контракту).
    veh_diff = int(np.abs(rgb.astype(int) - composite_u8.astype(int))[mask_bin].max()) \
        if mask_bin.any() else 0

    # Camera-realism degradation — на весь кадр, щоб зшити техніку й фон.
    deg_cfg = diffusion_cfg.get("degradation", {}) or {}
    degrade_on = bool(deg_cfg.get("enabled", False))
    degrade_stats: dict[str, Any] = {"enabled": degrade_on}
    if degrade_on:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seed = int(meta.get("seed", 0)) + int(deg_cfg.get("seed_offset", 7000))
        composite_u8, applied = _apply_camera_realism(composite_u8, deg_cfg, seed)
        degrade_stats.update(applied)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = Image.fromarray(composite_u8)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        out_img.save(out_path, quality=92)
    else:
        out_img.save(out_path)

    # Контракт: byte-identity лише коли НІ relight, НІ degradation (clean proof path).
    harmonized = relight_on and strength > 0
    if assert_pixel_identity and not degrade_on:
        if harmonized:
            # Техніка свідомо змінена — логуємо drift, не падаємо.
            relight_stats["veh_pixel_drift"] = veh_diff
        else:
            assert veh_diff == 0, (
                f"vehicle pixels modified (max diff={veh_diff}) when relight=OFF "
                f"and degradation=OFF — contract violation"
            )
    elif degrade_on:
        relight_stats["veh_pixel_drift_pre_degradation"] = veh_diff

    return {
        "image_size": [H, W],
        "vehicle_px": int(mask_bin.sum()),
        "bg_px": int((~mask_bin).sum()),
        "relight": relight_stats,
        "degradation": degrade_stats,
    }
