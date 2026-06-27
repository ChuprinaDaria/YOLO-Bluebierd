"""Prompt builder для Flux depth-conditioned inpaint stage.

Convention: azimuth 0° = East (+X у Blender world space). 8-way cardinal.
Реальні поля metadata пише `render_runner.py` у sidecar JSON.

Ціль промпту — НЕ кінематографічний рендер, а «погана» зйомка з дрона:
EO-сенсор, природне світло, легкий шум/компресія, top-down масштаб.
Тому базовий template доповнюється:
  * landscape-cue   — ставить техніку на дорогу/колію, а не «посеред поля»;
  * scale-cue       — підказує справжній aerial масштаб рослинності;
  * camera-cue      — amateur drone / EO sensor look замість cinematic.
Все можна перевизначити з cfg (`prompt_template`, `landscape_cues`,
`scale_cue`, `camera_cue`, `negative_prompt`), значення нижче — fallback.
"""

from __future__ import annotations


_CARDINAL_8 = ("east", "north-east", "north", "north-west",
               "west", "south-west", "south", "south-east")

# Landscape → де саме стоїть/їде техніка. Ключове для проблеми «посеред поля»:
# для road-сцен явно просимо дорогу/колію ПІД технікою, а не навколо.
_DEFAULT_LANDSCAPE_CUES = {
    "dirt_road": (
        "the vehicle is driving along a narrow dirt road, "
        "two parallel tire ruts in packed earth running directly beneath and "
        "ahead of the vehicle, dusty unpaved track"
    ),
    "field": (
        "the vehicle sits on a faint farm track crossing an open field, "
        "flattened grass and tire marks under the vehicle"
    ),
    "forest_belt": (
        "the vehicle is on a dirt track beside a tree line / forest shelterbelt, "
        "track running under the vehicle"
    ),
}

# Масштаб: Flux інакше малює траву/кущі «з рівня очей». Просимо top-down дрібність.
_DEFAULT_SCALE_CUE = (
    "seen straight from above at high altitude, everything at true aerial scale, "
    "vegetation appears as tiny fine texture, no large foreground objects"
)

# Камера: «погана» зйомка з дрона, НЕ cinematic. Терміни, які FLUX точно розуміє
# (grainy / surveillance / low-resolution / jpeg artifacts), а не «EO sensor».
_DEFAULT_CAMERA_CUE = (
    "low-resolution aerial surveillance footage, grainy telephoto drone shot, "
    "overcast flat lighting, slightly soft focus, visible sensor noise and "
    "jpeg compression artifacts, muted desaturated colors, "
    "candid unedited photo, amateur photo"
)

_DEFAULT_NEGATIVE = (
    "person, soldier, vehicle, tank, truck, car, motorcycle, building, "
    "road sign, text, watermark, ui, "
    # анти-«пластик» / анти-CGI / анти-кіно
    "3d render, cgi, render, video game, unreal engine, octane, blender, "
    "plastic, glossy, smooth plastic surface, airbrushed, waxy, "
    "cinematic, dramatic lighting, golden hour, lens flare, bokeh, depth of field, "
    "hdr, oversaturated, vivid, overprocessed, beautiful, masterpiece, artstation, "
    "oversharpened, illustration, painting, blurry, low quality"
)


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

    Повертає (positive, negative). До positive дочіпляються landscape-cue,
    scale-cue і camera-cue — щоб техніка стояла на дорозі, масштаб був aerial,
    а вигляд — як зйомка з дрона, а не кінокадр.
    """
    template = diffusion_cfg["prompt_template"]
    landscape_key = metadata["landscape"]
    positive = template.format(
        landscape=landscape_key.replace("_", " "),
        season=metadata["season"].replace("_", " "),
        weather=metadata["weather"],
        sun_cardinal=metadata["sun_cardinal"],
        sun_elevation_deg=metadata["sun_elevation_deg"],
        altitude_m=metadata["altitude_m"],
        view_angle_deg=metadata["view_angle_deg"],
    ).strip()

    cues: list[str] = []
    landscape_cues = diffusion_cfg.get("landscape_cues", _DEFAULT_LANDSCAPE_CUES)
    cue = landscape_cues.get(landscape_key)
    if cue:
        cues.append(cue.strip())
    scale_cue = diffusion_cfg.get("scale_cue", _DEFAULT_SCALE_CUE)
    if scale_cue:
        cues.append(scale_cue.strip())
    camera_cue = diffusion_cfg.get("camera_cue", _DEFAULT_CAMERA_CUE)
    if camera_cue:
        cues.append(camera_cue.strip())

    if cues:
        positive = positive.rstrip(". ") + ", " + ", ".join(cues)

    negative = diffusion_cfg.get("negative_prompt", _DEFAULT_NEGATIVE).strip()
    return positive, negative
