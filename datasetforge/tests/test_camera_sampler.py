"""Камера sampler — grid дekartовий продукт, round-robin рівномірний."""

from datasetforge.engine.camera_sampler import (
    CameraSample, build_grid, sample_random, sample_round_robin,
)


def test_grid_cartesian_product():
    grid = build_grid([200, 1000], [10, 20, 30])
    assert len(grid) == 6
    assert CameraSample(distance_m=200, view_angle_deg=10) in grid
    assert CameraSample(distance_m=1000, view_angle_deg=30) in grid


def test_round_robin_uniform():
    grid = build_grid([200, 500], [10, 30])
    samples = sample_round_robin(grid, 8)
    assert len(samples) == 8
    # 4 точки grid × 2 повних кола = по 2 кожна
    counts: dict = {}
    for s in samples:
        key = (s.distance_m, s.view_angle_deg)
        counts[key] = counts.get(key, 0) + 1
    assert all(c == 2 for c in counts.values())


def test_random_seeded():
    grid = build_grid([200, 1000], [10, 20, 30])
    a = sample_random(grid, 10, seed=42)
    b = sample_random(grid, 10, seed=42)
    assert a == b
