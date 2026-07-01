"""Stratifier altitude × view-angle × hfov.

Будує плоский список (altitude, view_angle, hfov) комбінацій з YAML config,
фільтрує комбінації, де ціль фізично занадто мала (preflight pixel budget),
вибирає round-robin або random для кожного кадру.

Схема камери — від РЕАЛЬНОГО дрона: робоча висота 150-200 м, крейсерська ~300 м,
ширококутна камера HFOV 92° або 112° (два апаратні варіанти). distance (line-of-sight)
— похідна величина: d = altitude / sin(view_angle).

Pure-Python. bpy не потрібен.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Мінімальна елевація — та сама, що clamp у scene_builder (крен нижче 5° не літаємо).
MIN_ELEV_DEG = 5.0

# bbox min-side ≈ ширина техніки; ширина ≈ 0.4 × довжини (Tigr 2.3/5.7, БТР 2.9/7.65).
WIDTH_RATIO = 0.4


@dataclass
class CameraSample:
    altitude_m: float        # висота польоту над ціллю (реальний параметр дрона)
    view_angle_deg: float    # елевація camera над горизонтом (90 = надір)
    hfov_deg: float          # горизонтальний FOV (92 або 112 — два варіанти камери)

    @property
    def distance_m(self) -> float:
        """3D line-of-sight камера→ціль, похідна від висоти і кута."""
        return self.altitude_m / math.sin(math.radians(max(self.view_angle_deg,
                                                           MIN_ELEV_DEG)))


def px_per_meter(sample: CameraSample, image_w: int) -> float:
    """Скільки пікселів у кадрі займає 1 метр на дистанції цілі."""
    footprint_m = 2.0 * sample.distance_m * math.tan(math.radians(sample.hfov_deg) / 2)
    return image_w / footprint_m


def estimate_target_px(
    sample: CameraSample,
    image_w: int,
    target_size_m: float,
) -> tuple[float, float]:
    """(довжина_px, оцінка_мін_сторони_px) цілі для комбінації камери.

    Мін-сторона bbox ≈ ширина техніки незалежно від ракурсу (oblique додає
    вертикальну проєкцію, але горизонтальна ≥ ширини) — консервативний lower bound.
    """
    ppm = px_per_meter(sample, image_w)
    return target_size_m * ppm, target_size_m * WIDTH_RATIO * ppm


def build_grid(
    altitudes: list[float],
    view_angles: list[float],
    hfovs: list[float],
) -> list[CameraSample]:
    """Декартів продукт altitudes × view_angles × hfovs."""
    return [
        CameraSample(altitude_m=a, view_angle_deg=v, hfov_deg=f)
        for a in altitudes
        for v in view_angles
        for f in hfovs
    ]


def build_grid_from_config(cam_cfg: dict) -> list[CameraSample]:
    """Грід із YAML `camera:` блоку. Нова схема — altitude_m; legacy — distance_m.

    hfov_deg може бути скаляром або списком (92/112 — два варіанти камери).
    Legacy distance_m: кожна (d, θ) пара конвертується у altitude = d·sin(θ),
    тому distance_m property повертає оригінальне d.
    """
    hfovs = cam_cfg["hfov_deg"]
    if not isinstance(hfovs, (list, tuple)):
        hfovs = [hfovs]
    hfovs = [float(f) for f in hfovs]
    view_angles = [float(v) for v in cam_cfg["view_angle_deg"]]

    if "altitude_m" in cam_cfg:
        altitudes = [float(a) for a in cam_cfg["altitude_m"]]
        return build_grid(altitudes, view_angles, hfovs)

    if "distance_m" in cam_cfg:
        return [
            CameraSample(
                altitude_m=float(d) * math.sin(math.radians(max(v, MIN_ELEV_DEG))),
                view_angle_deg=v,
                hfov_deg=f,
            )
            for d in cam_cfg["distance_m"]
            for v in view_angles
            for f in hfovs
        ]

    raise KeyError("camera config must have altitude_m (new) or distance_m (legacy)")


def filter_viable(
    grid: list[CameraSample],
    image_w: int,
    target_size_m: float,
    min_side_px: float,
) -> tuple[list[CameraSample], list[tuple[CameraSample, float]]]:
    """Preflight pixel budget: викидає комбінації, де ціль гарантовано занадто мала.

    Критерій ТОЙ САМИЙ, що post-render min-side фільтр bbox — інакше рендеримо
    кадри, які одразу підуть у discard (змарнований GPU-час).
    Повертає (viable, rejected) — rejected з оцінкою min-side для логу.
    """
    viable: list[CameraSample] = []
    rejected: list[tuple[CameraSample, float]] = []
    for s in grid:
        _, est_min = estimate_target_px(s, image_w, target_size_m)
        if est_min >= min_side_px:
            viable.append(s)
        else:
            rejected.append((s, est_min))
    return viable, rejected


def sample_round_robin(
    grid: list[CameraSample],
    n: int,
) -> list[CameraSample]:
    """Рівномірне покриття grid: проходить по колу `n` разів."""
    return [grid[i % len(grid)] for i in range(n)]


def sample_random(
    grid: list[CameraSample],
    n: int,
    seed: int = 0,
) -> list[CameraSample]:
    rng = random.Random(seed)
    return [rng.choice(grid) for _ in range(n)]
