"""Stage 5 (shared) — «шліфування поверх»: камерна деградація через albumentations.

Зерно/шум/туман/JPEG/motion-blur — це post-processing, НЕ робота генеративної моделі.
Той самий augmentation робить датасет ближчим до реального дронового відео.

ВАЖЛИВО: усі трансформи тут — pixel-only (не рухають геометрію), тому **bbox/мітки
не змінюються**. Накладається на ВЕСЬ composite-кадр (і техніку, і фон) — це зшиває
їх у єдину «камеру».

API: albumentations 2.x (pinned >=2.0,<2.1). Назви параметрів звірені з 2.0.8:
  ImageCompression(quality_range=...)   RandomFog(fog_coef_range=, alpha_coef=)
  GaussNoise(std_range=...)  ISONoise(color_shift=, intensity=)
  Downscale(scale_range=...)  ChromaticAberration(primary_distortion_limit=, ...)
Seed → A.Compose(..., seed=seed) для відтворюваності.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Дефолти на випадок голого `polish: {enabled: true}`. Кожен блок вмикається лише
# якщо присутній у cfg (або тут) і має p>0. Значення — стартові для drone-look.
_DEFAULTS: dict[str, Any] = {
    "iso_noise": {"p": 0.7, "color_shift": [0.01, 0.05], "intensity": [0.1, 0.5]},
    "gauss_noise": {"p": 0.35, "std_range": [0.02, 0.07]},  # частка від 255
    "motion_blur": {"p": 0.3, "blur_limit": [3, 7]},
    "defocus": {"p": 0.15, "radius": [1, 3]},
    "fog": {"p": 0.25, "fog_coef_range": [0.05, 0.25], "alpha_coef": 0.08},
    "brightness_contrast": {"p": 0.5, "brightness": 0.12, "contrast": 0.12},
    "chromatic": {"p": 0.3, "primary": 0.02, "secondary": 0.01},
    "downscale": {"p": 0.2, "scale_range": [0.6, 0.9]},
    "jpeg": {"p": 0.9, "quality": [60, 85]},  # майже завжди — головний «дрон/телефон» tell
}


def _tuple(v, fallback):
    if v is None:
        return fallback
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (v, v)


def build_polish(polish_cfg: dict):
    """Будує A.Compose зі списку увімкнених трансформів. None якщо все вимкнено.

    JPEG ставимо ОСТАННІМ (реальний кодек-порядок). Compose без bbox_params —
    геометрія не чіпається, мітки лишаються зовні незмінними.
    """
    import albumentations as A

    cfg = {**_DEFAULTS, **{k: v for k, v in (polish_cfg or {}).items()
                           if k in _DEFAULTS and v is not None}}
    # Дозволяємо вимкнути окремий блок передавши false/{enabled:false}/p<=0.
    def on(name: str) -> dict | None:
        v = polish_cfg.get(name, _DEFAULTS[name]) if polish_cfg else _DEFAULTS[name]
        if v is False or v is None:
            return None
        d = {**_DEFAULTS[name], **(v if isinstance(v, dict) else {})}
        if d.get("enabled") is False or float(d.get("p", 1.0)) <= 0:
            return None
        return d

    tf = []

    if (d := on("iso_noise")):
        tf.append(A.ISONoise(color_shift=_tuple(d["color_shift"], (0.01, 0.05)),
                             intensity=_tuple(d["intensity"], (0.1, 0.5)),
                             p=float(d["p"])))
    if (d := on("gauss_noise")):
        tf.append(A.GaussNoise(std_range=_tuple(d["std_range"], (0.02, 0.07)),
                               p=float(d["p"])))
    if (d := on("motion_blur")):
        tf.append(A.MotionBlur(blur_limit=_tuple(d["blur_limit"], (3, 7)),
                               p=float(d["p"])))
    if (d := on("defocus")):
        tf.append(A.Defocus(radius=_tuple(d["radius"], (1, 3)), p=float(d["p"])))
    if (d := on("fog")):
        tf.append(A.RandomFog(fog_coef_range=_tuple(d["fog_coef_range"], (0.05, 0.25)),
                              alpha_coef=float(d.get("alpha_coef", 0.08)),
                              p=float(d["p"])))
    if (d := on("brightness_contrast")):
        tf.append(A.RandomBrightnessContrast(
            brightness_limit=float(d["brightness"]),
            contrast_limit=float(d["contrast"]), p=float(d["p"])))
    if (d := on("chromatic")):
        tf.append(A.ChromaticAberration(
            primary_distortion_limit=float(d["primary"]),
            secondary_distortion_limit=float(d["secondary"]), p=float(d["p"])))
    if (d := on("downscale")):
        tf.append(A.Downscale(scale_range=_tuple(d["scale_range"], (0.6, 0.9)),
                              p=float(d["p"])))
    if (d := on("jpeg")):
        tf.append(A.ImageCompression(quality_range=_tuple(d["quality"], (60, 85)),
                                     p=float(d["p"])))

    if not tf:
        return None
    return A.Compose(tf)


def polish_one(in_path: Path, out_path: Path, polish_cfg: dict,
               seed: int = 0) -> dict[str, Any]:
    """Накласти полиш на один composite-кадр. bbox/мітки не чіпаються.

    Якщо polish вимкнено — просто копіює (re-save) кадр без змін.
    """
    img = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8)
    polish_on = bool((polish_cfg or {}).get("enabled", False))

    applied = []
    if polish_on:
        pipe = build_polish(polish_cfg)
        if pipe is not None:
            # seed на Compose → відтворюваний результат для того ж кадру.
            try:
                pipe.set_random_seed(int(seed) & 0xFFFFFFFF)
            except Exception:
                pass
            img = pipe(image=img)["image"]
            applied = [t.__class__.__name__ for t in pipe.transforms]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = Image.fromarray(img)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        out_img.save(out_path, quality=95)
    else:
        out_img.save(out_path)

    return {"enabled": polish_on, "transforms": applied, "seed": int(seed)}
