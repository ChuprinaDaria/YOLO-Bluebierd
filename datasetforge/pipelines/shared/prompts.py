"""Prompt builder для diffusion background-стейджів (shared).

Convention: azimuth 0° = East (+X у Blender world space). 8-way cardinal.
Реальні поля metadata пише `render_runner.py` у sidecar JSON.

Два режими (`mode`):
  * "describe"  — для inpaint-моделей (FLUX-Depth, FLUX-Fill): описовий промпт сцени.
  * "instruct"  — для instruction-edit (Qwen-Image-Edit-2509): інструкція
    «лиши техніку, перероби лише фон». Qwen не має mask — техніку все одно
    заморозить Stage 4 composite, але інструкція допомагає моделі не чіпати її.

Ціль вигляду — НЕ кінематографічний рендер, а «погана» зйомка з дрона:
EO-сенсор, природне світло, top-down масштаб. Тому описова частина доповнюється
landscape-cue (техніка на дорозі, не «посеред поля»), scale-cue і camera-cue.
Все можна перевизначити з cfg.
"""

from __future__ import annotations


_CARDINAL_8 = ("east", "north-east", "north", "north-west",
               "west", "south-west", "south", "south-east")

# Landscape → де саме стоїть/їде техніка. Ключове проти «посеред поля».
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

_DEFAULT_SCALE_CUE = (
    "seen straight from above at high altitude, everything at true aerial scale, "
    "vegetation appears as tiny fine texture, no large foreground objects"
)

_DEFAULT_CAMERA_CUE = (
    "low-resolution aerial surveillance footage, grainy telephoto drone shot, "
    "overcast flat lighting, slightly soft focus, visible sensor noise and "
    "jpeg compression artifacts, muted desaturated colors, "
    "candid unedited photo, amateur photo"
)

_DEFAULT_NEGATIVE = (
    "tank, military vehicle, armored vehicle, APC, IFV, MBT, T-72, T-80, T-90, "
    "vehicle on road, multiple vehicles, second vehicle, duplicate vehicle, "
    "twin vehicle, mirrored vehicle, extra vehicle, ghost vehicle, "
    "person, soldier, vehicle, truck, car, motorcycle, building, road sign, "
    "text, watermark, ui, 3d render, cgi, render, video game, unreal engine, octane, "
    "blender, plastic, glossy, waxy, airbrushed, cinematic, dramatic lighting, "
    "golden hour, lens flare, bokeh, depth of field, hdr, oversaturated, vivid, "
    "overprocessed, masterpiece, artstation, oversharpened, illustration, painting, "
    "blurry, low quality"
)


def azimuth_to_cardinal(deg: float) -> str:
    """0° = east (Blender +X). 8-way @ 22.5° boundaries."""
    deg = deg % 360.0
    # Зміщуємо на +22.5 щоб east bucket центрувався на 0°.
    idx = int(((deg + 22.5) % 360) // 45)
    return _CARDINAL_8[idx]


def _scene_description(metadata: dict, diffusion_cfg: dict) -> str:
    """Описова частина (template + landscape/scale/camera cues), спільна для обох mode."""
    template = diffusion_cfg["prompt_template"]
    landscape_key = metadata["landscape"]
    scene = template.format(
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
        scene = scene.rstrip(". ") + ", " + ", ".join(cues)
    return scene


def build_prompt(metadata: dict, diffusion_cfg: dict,
                 mode: str = "describe") -> tuple[str, str]:
    """Повертає (positive, negative).

    mode="describe" → positive = опис сцени (inpaint моделі домалюють у masked зоні).
    mode="instruct" → positive = інструкція «лиши техніку, перероби фон: <опис>».
    """
    scene = _scene_description(metadata, diffusion_cfg)

    if mode == "instruct":
        # Vehicle area вже erased у load_model.edit_one (cv2.inpaint) — Qwen бачить
        # clean landscape. Lead тепер веде з "empty terrain", щоб модель не намалювала
        # vehicle з prompt context ("drone reconnaissance" може імплікувати targets).
        lead = diffusion_cfg.get(
            "instruct_lead",
            "Generate a clean top-down aerial drone photo of an empty rural landscape. "
            "No vehicles, no military equipment, no people, no traffic. Just terrain. "
            "Scene: ",
        )
        positive = lead.strip().rstrip(":") + ": " + scene
    else:
        positive = scene

    negative = diffusion_cfg.get("negative_prompt", _DEFAULT_NEGATIVE).strip()
    return positive, negative
