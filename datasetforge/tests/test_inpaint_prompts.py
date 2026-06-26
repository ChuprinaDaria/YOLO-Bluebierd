"""Unit tests для datasetforge.pipelines.inpaint.prompts.

Pure-Python (без torch/cv2/Flux) — крутиться на будь-чому.
"""

from __future__ import annotations

import pytest

from datasetforge.pipelines.inpaint.prompts import (
    azimuth_to_cardinal,
    build_prompt,
)


class TestAzimuthToCardinal:
    """Convention: 0° = east (Blender +X), 8-way @ 22.5° boundaries."""

    @pytest.mark.parametrize("deg,expected", [
        (0.0, "east"),
        (22.4, "east"),
        (22.5, "north-east"),
        (45.0, "north-east"),
        (67.5, "north"),
        (90.0, "north"),
        (135.0, "north-west"),
        (180.0, "west"),
        (225.0, "south-west"),
        (270.0, "south"),
        (315.0, "south-east"),
        (359.0, "east"),
        (360.0, "east"),
        (720.0, "east"),
        (-90.0, "south"),
        (-180.0, "west"),
    ])
    def test_boundaries(self, deg, expected):
        assert azimuth_to_cardinal(deg) == expected


class TestBuildPrompt:
    @pytest.fixture
    def diff_cfg(self):
        return {
            "prompt_template": (
                "aerial drone reconnaissance photo of {landscape} in {season}, "
                "{weather}, low {sun_cardinal} sun at {sun_elevation_deg:.0f} "
                "degrees elevation, from {altitude_m:.0f}m altitude at "
                "{view_angle_deg:.0f} degrees oblique angle, photorealistic"
            ),
            "negative_prompt": "person, soldier, text, watermark, blurry",
        }

    @pytest.fixture
    def metadata(self):
        return {
            "landscape": "forest_belt",
            "season": "autumn_mud",
            "weather": "rain",
            "sun_cardinal": "south-west",
            "sun_elevation_deg": 35.7,
            "altitude_m": 247.3,
            "view_angle_deg": 22.0,
        }

    def test_template_fills_all_keys(self, diff_cfg, metadata):
        pos, neg = build_prompt(metadata, diff_cfg)
        assert "forest belt" in pos                 # underscore → space
        assert "autumn mud" in pos
        assert "rain" in pos
        assert "south-west sun" in pos
        assert "36 degrees elevation" in pos        # int-formatted (35.7 → 36)
        assert "247m altitude" in pos
        assert "22 degrees oblique" in pos
        assert "{" not in pos                       # no unfilled placeholders
        assert neg.strip() == diff_cfg["negative_prompt"].strip()

    def test_negative_passthrough(self, diff_cfg, metadata):
        _, neg = build_prompt(metadata, diff_cfg)
        assert "person" in neg
        assert "watermark" in neg

    def test_missing_key_raises(self, diff_cfg):
        bad = {"landscape": "field"}  # навмисно неповне
        with pytest.raises(KeyError):
            build_prompt(bad, diff_cfg)

    def test_underscore_strip_only_in_replace_fields(self, diff_cfg, metadata):
        """sun_cardinal вже "south-west" — НЕ підмінюємо дефіс."""
        pos, _ = build_prompt(metadata, diff_cfg)
        assert "south-west" in pos  # дефіс зберігся
