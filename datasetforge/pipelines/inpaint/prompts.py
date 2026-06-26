"""Prompt builder для Flux depth-conditioned inpaint stage.

Convention: azimuth 0° = East (+X у Blender world space). 8-way cardinal.
Реальні поля metadata пише `render_runner.py` у sidecar JSON.
"""

from __future__ import annotations


_CARDINAL_8 = ("east", "north-east", "north", "north-west",
               "west", "south-west", "south", "south-east")


def azimuth_to_cardinal(deg: float) -> str:
    """0° = east (Blender +X). 8-way @ 22.5° boundaries."""
    deg = deg % 360.0
    # Зміщуємо на +22.5 щоб east bucket центрувався на 0°.
    idx = int(((deg + 22.5) % 360) // 45)
    return _CARDINAL_8[idx]


def build_prompt(metadata: dict, diffusion_cfg: dict) -> tuple[str, str]:
    """Заповнює prompt_template з cfg значеннями з per-frame metadata.

    Очікувані ключі metadata (з render_runner sidecar):
      landscape, season, weather, altitude_m, view_angle_deg,
      sun_cardinal, sun_elevation_deg.
    """
    template = diffusion_cfg["prompt_template"]
    positive = template.format(
        landscape=metadata["landscape"].replace("_", " "),
        season=metadata["season"].replace("_", " "),
        weather=metadata["weather"],
        sun_cardinal=metadata["sun_cardinal"],
        sun_elevation_deg=metadata["sun_elevation_deg"],
        altitude_m=metadata["altitude_m"],
        view_angle_deg=metadata["view_angle_deg"],
    ).strip()
    negative = diffusion_cfg["negative_prompt"].strip()
    return positive, negative
