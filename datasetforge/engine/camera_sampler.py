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


def sample_stratified(
    grid: list[CameraSample],
    n: int,
    seed: int = 0,
) -> list[CameraSample]:
    """Stratified pick: гарантує coverage по distance bins.

    `random.choice(grid)` з малим n може degenerate (всі picks у одному
    distance bucket — seed=42 + 5 picks дало all-distance=300). Stratified:
    розкладає n рівномірно по unique distances, у кожному bucket рандомно
    тягне (distance, angle) комбінацію.
    """
    rng = random.Random(seed)
    buckets: dict[float, list[CameraSample]] = {}
    for s in grid:
        buckets.setdefault(s.distance_m, []).append(s)
    distances = sorted(buckets.keys())
    samples: list[CameraSample] = []
    for i in range(n):
        d = distances[i % len(distances)]
        samples.append(rng.choice(buckets[d]))
    rng.shuffle(samples)
    return samples
