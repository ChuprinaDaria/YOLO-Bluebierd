"""Unit tests для datasetforge.pipelines.inpaint.composite.

Без Flux/GPU — синтетичні RGB/depth/mask/normals/metadata через tmp_path
fixture. Валідує контракт frozen-vehicle + bg replacement + relight tolerance.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from datasetforge.pipelines.inpaint.composite import (
    _sun_vec_from_meta,
    composite_one,
)


@pytest.fixture
def workspace(tmp_path):
    """Synthetic Stage 1 outputs.

    Vehicle area = 96×96 px у центрі 256×256. RGB має solid colour там
    (легко перевірити що не зміщений). AI bg перемальовує цю область
    у "garbage" — якщо composite дозволить bg пройти на vehicle, тест зловить.
    """
    H, W = 256, 256

    rgb = np.random.RandomState(42).randint(0, 256, (H, W, 3), dtype=np.uint8)
    rgb[80:176, 80:176] = [200, 100, 50]   # vehicle = solid orange
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)    # PNG, lossless

    ai_bg = np.full((H, W, 3), [50, 200, 200], dtype=np.uint8)
    ai_bg[80:176, 80:176] = [10, 10, 10]   # AI bg "spilled into" vehicle
    ai_bg_path = tmp_path / "ai_bg.png"
    Image.fromarray(ai_bg).save(ai_bg_path)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[80:176, 80:176] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    # Normals: constant up-pointing (Z=+1) — render_runner encoding
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    normals_u16 = np.clip(((normals + 1.0) * 0.5) * 65535.0,
                          0, 65535).astype(np.uint16)
    normals_path = tmp_path / "normals.png"
    cv2.imwrite(str(normals_path), normals_u16)

    meta = {
        "seed": 42,
        "sun_azimuth_deg": 225.0,   # south-west
        "sun_elevation_deg": 30.0,
        "landscape": "field",
        "season": "summer",
        "weather": "clear",
        "sun_cardinal": "south-west",
        "altitude_m": 200.0,
        "view_angle_deg": 20.0,
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))

    return {
        "rgb_path": rgb_path,
        "ai_bg_path": ai_bg_path,
        "mask_path": mask_path,
        "normals_path": normals_path,
        "meta_path": meta_path,
        "rgb": rgb,
        "ai_bg": ai_bg,
        "mask": mask,
        "H": H,
        "W": W,
    }


def _bool_mask(workspace):
    raw = cv2.imread(str(workspace["mask_path"]), cv2.IMREAD_GRAYSCALE)
    return raw >= 128


class TestPixelIdentityContract:
    """Контракт: relight OFF → composite[mask] байт-у-байт == rgb[mask]."""

    def test_internal_assertion_passes(self, workspace, tmp_path):
        cfg = {"relight": {"enabled": False, "strength": 0.0}}
        out_path = tmp_path / "composite.png"
        stats = composite_one(
            rgb_path=workspace["rgb_path"],
            ai_bg_path=workspace["ai_bg_path"],
            mask_path=workspace["mask_path"],
            normals_path=workspace["normals_path"],
            meta_path=workspace["meta_path"],
            out_path=out_path,
            diffusion_cfg=cfg,
            assert_pixel_identity=True,
        )
        assert stats["relight"]["enabled"] is False
        assert stats["vehicle_px"] == 96 * 96
        assert stats["bg_px"] == 256 * 256 - 96 * 96

    def test_external_recheck_vehicle_byte_identical(self, workspace, tmp_path):
        cfg = {"relight": {"enabled": False, "strength": 0.0}}
        out_path = tmp_path / "composite.png"
        composite_one(
            rgb_path=workspace["rgb_path"],
            ai_bg_path=workspace["ai_bg_path"],
            mask_path=workspace["mask_path"],
            normals_path=workspace["normals_path"],
            meta_path=workspace["meta_path"],
            out_path=out_path,
            diffusion_cfg=cfg,
            assert_pixel_identity=False,
        )
        comp = np.array(Image.open(out_path))
        mask_bool = _bool_mask(workspace)
        diff = int(np.abs(workspace["rgb"].astype(int)
                          - comp.astype(int))[mask_bool].max())
        assert diff == 0, f"vehicle pixel drift {diff}"

    def test_bg_pixels_from_ai_bg(self, workspace, tmp_path):
        cfg = {"relight": {"enabled": False, "strength": 0.0}}
        out_path = tmp_path / "composite.png"
        composite_one(
            rgb_path=workspace["rgb_path"],
            ai_bg_path=workspace["ai_bg_path"],
            mask_path=workspace["mask_path"],
            normals_path=workspace["normals_path"],
            meta_path=workspace["meta_path"],
            out_path=out_path,
            diffusion_cfg=cfg,
            assert_pixel_identity=False,
        )
        comp = np.array(Image.open(out_path))
        mask_bool = _bool_mask(workspace)
        diff_bg = int(np.abs(workspace["ai_bg"].astype(int)
                             - comp.astype(int))[~mask_bool].max())
        assert diff_bg == 0, f"bg pixel drift {diff_bg} — composite not using ai_bg"


class TestRelightOn:
    """relight ON → vehicle модифікований у межах tolerance."""

    def test_relight_within_tolerance(self, workspace, tmp_path):
        cfg = {"relight": {"enabled": True, "strength": 0.3}}
        out_path = tmp_path / "composite.png"
        stats = composite_one(
            rgb_path=workspace["rgb_path"],
            ai_bg_path=workspace["ai_bg_path"],
            mask_path=workspace["mask_path"],
            normals_path=workspace["normals_path"],
            meta_path=workspace["meta_path"],
            out_path=out_path,
            diffusion_cfg=cfg,
            assert_pixel_identity=True,
        )
        assert stats["relight"]["enabled"] is True
        # Internal assertion прошла → drift ≤ int(255 * 0.3) + 1 = 77

    def test_relight_disabled_default(self, workspace, tmp_path):
        cfg = {}  # порожньо — relight default = OFF
        out_path = tmp_path / "composite.png"
        stats = composite_one(
            rgb_path=workspace["rgb_path"],
            ai_bg_path=workspace["ai_bg_path"],
            mask_path=workspace["mask_path"],
            normals_path=workspace["normals_path"],
            meta_path=workspace["meta_path"],
            out_path=out_path,
            diffusion_cfg=cfg,
            assert_pixel_identity=True,
        )
        assert stats["relight"]["enabled"] is False


class TestSunVecFromMeta:
    """Convention: azimuth 0° = +X (east), elevation = angle above horizon."""

    def test_east_horizon(self):
        v = _sun_vec_from_meta({"sun_azimuth_deg": 0.0, "sun_elevation_deg": 0.0})
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-6)

    def test_north_horizon(self):
        v = _sun_vec_from_meta({"sun_azimuth_deg": 90.0, "sun_elevation_deg": 0.0})
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-6)

    def test_zenith(self):
        v = _sun_vec_from_meta({"sun_azimuth_deg": 0.0, "sun_elevation_deg": 90.0})
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-6)

    def test_unit_length(self):
        v = _sun_vec_from_meta({"sun_azimuth_deg": 67.0, "sun_elevation_deg": 33.0})
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6
