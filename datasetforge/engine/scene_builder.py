"""Скелет Blender scene builder.

Запускається через `blender --background --python -- scene_builder.py <config>`.

Робить (TODO):
1. Завантажує 3D-модель з `assets/models/<class>/<variant>.blend` через bpy.ops.import.
2. Додає background (HDR панораму або plane з drone-frame texture).
3. Встановлює камеру з висотою/кутом/HFOV з конфігу.
4. Світло: sun з кутом який матчить time_of_day і season.
5. Повертає bpy.context.scene готовий для рендеру.

Структура камери:
    camera_z = altitude_m (метри в Blender ≈ meters)
    camera_pitch = -view_angle_deg (negative = look down)
    camera_lens = focal length з hfov_deg + sensor width

TODO: bpy imports тільки під Blender. Тут — інтерфейс.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraSpec:
    altitude_m: float
    view_angle_deg: float
    hfov_deg: float
    sensor_width_mm: float = 35.0

    @property
    def focal_mm(self) -> float:
        import math

        return self.sensor_width_mm / (2 * math.tan(math.radians(self.hfov_deg) / 2))


@dataclass
class SceneRequest:
    class_name: str
    model_path: Path
    background_path: Path
    camera: CameraSpec
    season: str
    landscape: str
    weather: str
    seed: int


def build_scene(req: SceneRequest) -> None:
    """TODO: викликає bpy. Зараз — placeholder."""
    raise NotImplementedError("Implement under Blender Python (bpy)")
