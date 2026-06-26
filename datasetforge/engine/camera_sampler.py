"""Stratifier altitude × view-angle.

Будує плоский список (altitude, view_angle) комбінацій з YAML config,
вибирає round-robin або random для кожного кадру.

Pure-Python. bpy не потрібен.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class CameraSample:
    distance_m: float        # пряма distance camera → vehicle (200-1000м)
    view_angle_deg: float    # елевація camera над горизонтом


def build_grid(
    distances: list[float],
    view_angles: list[float],
) -> list[CameraSample]:
    """Декартів продукт distances × view_angles."""
    return [
        CameraSample(distance_m=d, view_angle_deg=v)
        for d in distances
        for v in view_angles
    ]


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
