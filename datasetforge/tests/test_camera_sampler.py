"""Unit tests для datasetforge.engine.camera_sampler.

Pure-Python (без bpy) — крутиться на будь-чому. Числа звірені з
docs/pixel_budget.md (реальна камера: 150-300 м, HFOV 92°/112°, 1920px).
"""

from __future__ import annotations

import math

import pytest

from datasetforge.engine.camera_sampler import (
    CameraSample,
    build_grid,
    build_grid_from_config,
    estimate_target_px,
    filter_viable,
    px_per_meter,
    sample_round_robin,
)

REAL_CAM_CFG = {
    "altitude_m": [150, 200, 300],
    "view_angle_deg": [30, 45, 60, 90],
    "hfov_deg": [92, 112],
}


class TestCameraSample:
    def test_distance_nadir_equals_altitude(self):
        s = CameraSample(altitude_m=150, view_angle_deg=90, hfov_deg=92)
        assert s.distance_m == pytest.approx(150.0)

    def test_distance_oblique(self):
        # d = alt / sin(30°) = 2×alt
        s = CameraSample(altitude_m=150, view_angle_deg=30, hfov_deg=92)
        assert s.distance_m == pytest.approx(300.0)

    def test_low_angle_clamped_to_min_elev(self):
        # кут 0° не дає ділення на нуль — clamp до 5°
        s = CameraSample(altitude_m=150, view_angle_deg=0, hfov_deg=92)
        assert s.distance_m == pytest.approx(150.0 / math.sin(math.radians(5.0)))


class TestPixelBudget:
    def test_px_per_meter_92(self):
        # @92°, надір 150 м: 1920 / (2·150·tan46°) ≈ 6.18 px/м
        s = CameraSample(altitude_m=150, view_angle_deg=90, hfov_deg=92)
        assert px_per_meter(s, 1920) == pytest.approx(6.18, abs=0.05)

    def test_estimate_tigr_nadir_150(self):
        # Ціль 5.7 м: ~35px довжина / ~14px min-side (pixel_budget.md)
        s = CameraSample(altitude_m=150, view_angle_deg=90, hfov_deg=92)
        long_px, min_px = estimate_target_px(s, 1920, 5.7)
        assert long_px == pytest.approx(35, abs=1)
        assert min_px == pytest.approx(14, abs=1)

    def test_112_worse_than_92(self):
        s92 = CameraSample(altitude_m=150, view_angle_deg=90, hfov_deg=92)
        s112 = CameraSample(altitude_m=150, view_angle_deg=90, hfov_deg=112)
        assert px_per_meter(s112, 1920) < px_per_meter(s92, 1920)


class TestGrid:
    def test_cartesian_size(self):
        grid = build_grid([150, 200, 300], [30, 45, 60, 90], [92, 112])
        assert len(grid) == 24

    def test_from_config_new_schema(self):
        grid = build_grid_from_config(REAL_CAM_CFG)
        assert len(grid) == 24
        assert {s.hfov_deg for s in grid} == {92.0, 112.0}

    def test_from_config_scalar_hfov(self):
        grid = build_grid_from_config(
            {"altitude_m": [150], "view_angle_deg": [90], "hfov_deg": 92})
        assert len(grid) == 1

    def test_from_config_legacy_distance(self):
        # legacy distance_m: altitude = d·sin(θ), а distance_m property
        # повертає оригінальне d (roundtrip)
        grid = build_grid_from_config(
            {"distance_m": [1500], "view_angle_deg": [30], "hfov_deg": 6})
        assert len(grid) == 1
        assert grid[0].altitude_m == pytest.approx(750.0)
        assert grid[0].distance_m == pytest.approx(1500.0)

    def test_from_config_missing_keys_raises(self):
        with pytest.raises(KeyError):
            build_grid_from_config({"view_angle_deg": [30], "hfov_deg": 92})


class TestFilterViable:
    def test_real_drone_grid(self):
        # Реальні параметри: 14/24 комбінацій проходять min_side 6px
        # (крейсер 300 м виживає ТІЛЬКИ @92° при 60-90°).
        grid = build_grid_from_config(REAL_CAM_CFG)
        viable, rejected = filter_viable(grid, 1920, 5.7, 6)
        assert len(viable) == 14
        assert len(rejected) == 10
        cruise_112 = [s for s in viable
                      if s.altitude_m == 300 and s.hfov_deg == 112]
        assert cruise_112 == []

    def test_rejected_carry_estimate(self):
        grid = build_grid_from_config(REAL_CAM_CFG)
        _, rejected = filter_viable(grid, 1920, 5.7, 6)
        for sample, est_min in rejected:
            assert est_min < 6

    def test_old_640_render_would_reject_everything(self):
        # Регресія «крапок»: старий рендер 640 @ hfov 6°/1500-2700 м
        # не проходить власний поріг 10px min-side НІДЕ.
        legacy = build_grid_from_config(
            {"distance_m": [1500, 2100, 2700], "view_angle_deg": [10, 30, 90],
             "hfov_deg": 6})
        viable, _ = filter_viable(legacy, 640, 5.0, 10)
        assert viable == []


class TestSampling:
    def test_round_robin_covers_grid(self):
        grid = build_grid([150, 200], [45, 90], [92])
        samples = sample_round_robin(grid, 8)
        assert len(samples) == 8
        assert samples[:4] == grid  # повне коло перед повтором
        assert samples[4:] == grid
