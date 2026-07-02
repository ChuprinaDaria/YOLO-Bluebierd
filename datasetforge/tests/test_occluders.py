"""Unit tests для datasetforge.engine.occluders (pure-Python placement).

build_occluders потребує bpy — тут тестуємо тільки plan_occluders (геометрію).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from datasetforge.engine.occluders import OCCLUDER_CATEGORY_ID, plan_occluders


def _rng():
    return np.random.default_rng(42)


class TestPlanOccluders:
    def test_count(self):
        specs = plan_occluders((0, 0), (100, 0), _rng(), n=5, kinds=["tree"])
        assert len(specs) == 5

    def test_occluders_between_vehicle_and_camera(self):
        # камера на +X: оклюдери мають бути з боку камери (x>0), близько до цілі
        specs = plan_occluders((0, 0), (200, 0), _rng(), n=8,
                               kinds=["tree", "bush"], gap_m=(1.5, 6.0))
        for s in specs:
            # gap 1.5-6 м уздовж +X, боковий зсув ±3 → x у [1.5, 6], |y| ≤ ~3
            assert 1.0 < s.x < 7.0
            assert abs(s.y) < 4.0

    def test_lateral_axis_follows_camera_direction(self):
        # камера на +Y: боковий зсув має йти по X (перпендикуляр)
        specs = plan_occluders((0, 0), (0, 200), _rng(), n=8, kinds=["tree"])
        for s in specs:
            assert 1.0 < s.y < 7.0      # gap уздовж +Y
            assert abs(s.x) < 4.0       # боковий зсув по X

    def test_nadir_camera_no_division_by_zero(self):
        # камера строго над ціллю (той самий xy) — не має падати
        specs = plan_occluders((5, 5), (5, 5), _rng(), n=3, kinds=["bush"])
        assert len(specs) == 3
        for s in specs:
            assert math.isfinite(s.x) and math.isfinite(s.y)

    def test_kinds_respected(self):
        specs = plan_occluders((0, 0), (50, 0), _rng(), n=20, kinds=["net"])
        assert all(s.kind == "net" for s in specs)

    def test_trees_taller_than_bushes(self):
        trees = plan_occluders((0, 0), (50, 0), _rng(), n=30, kinds=["tree"])
        bushes = plan_occluders((0, 0), (50, 0), _rng(), n=30, kinds=["bush"])
        assert min(s.height_m for s in trees) > max(s.height_m for s in bushes)

    def test_all_grounded(self):
        specs = plan_occluders((0, 0), (50, 0), _rng(), n=10,
                               kinds=["tree", "bush", "net"])
        assert all(s.z == 0.0 for s in specs)

    def test_category_is_background(self):
        # оклюдери — не ціль, не потрапляють у vehicle-маску
        assert OCCLUDER_CATEGORY_ID == 0
