"""Unit tests для datasetforge.engine.bbox_extractor (pure-Python частина).

Ключова поведінка: coco_to_yolo звітує n_dropped — кількість bbox, викинутих
min-side фільтром. render_runner на основі цього ВІДКИДАЄ кадр цілком
(видима-але-нерозмічена техніка = отруйний false negative).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datasetforge.engine.bbox_extractor import (
    YoloBox,
    YoloObb,
    coco_to_yolo,
    mask_to_yolo_box,
    mask_to_yolo_obb,
    write_yolo_label,
    write_yolo_obb,
)


def _write_coco(tmp_path, boxes, image_w=1920, image_h=1080):
    coco = {
        "images": [{"id": 1, "file_name": "000000.jpg",
                    "width": image_w, "height": image_h}],
        "annotations": [
            {"id": k, "image_id": 1, "category_id": 7, "bbox": list(b),
             "iscrowd": 0}
            for k, b in enumerate(boxes)
        ],
    }
    p = tmp_path / "coco_annotations.json"
    p.write_text(json.dumps(coco), encoding="utf-8")
    return p


class TestCocoToYolo:
    def test_box_above_threshold_kept(self, tmp_path):
        p = _write_coco(tmp_path, [(100, 100, 35, 14)])
        [(fname, boxes, n_dropped)] = coco_to_yolo(p, 1920, 1080, min_side_px=6)
        assert fname == "000000.jpg"
        assert len(boxes) == 1
        assert n_dropped == 0
        b = boxes[0]
        assert b.cls == 7
        assert b.xc == pytest.approx((100 + 35 / 2) / 1920)
        assert b.w == pytest.approx(35 / 1920)

    def test_subthreshold_box_dropped_and_reported(self, tmp_path):
        # min side 4px < 6 → викинуто, n_dropped=1 — сигнал для discard кадру
        p = _write_coco(tmp_path, [(100, 100, 12, 4)])
        [(_, boxes, n_dropped)] = coco_to_yolo(p, 1920, 1080, min_side_px=6)
        assert boxes == []
        assert n_dropped == 1

    def test_mixed_boxes(self, tmp_path):
        p = _write_coco(tmp_path, [(100, 100, 35, 14), (500, 500, 10, 4)])
        [(_, boxes, n_dropped)] = coco_to_yolo(p, 1920, 1080, min_side_px=6)
        assert len(boxes) == 1
        assert n_dropped == 1

    def test_no_annotations_zero_dropped(self, tmp_path):
        # чесний порожній кадр (hard negative): 0 боксів, 0 dropped
        p = _write_coco(tmp_path, [])
        [(_, boxes, n_dropped)] = coco_to_yolo(p, 1920, 1080, min_side_px=6)
        assert boxes == []
        assert n_dropped == 0


class TestWriteYoloLabel:
    def test_writes_lines(self, tmp_path):
        out = tmp_path / "l.txt"
        write_yolo_label([YoloBox(cls=7, xc=0.5, yc=0.5, w=0.02, h=0.01)], out)
        assert out.read_text().strip() == "7 0.500000 0.500000 0.020000 0.010000"

    def test_empty_creates_empty_file(self, tmp_path):
        out = tmp_path / "l.txt"
        write_yolo_label([], out)
        assert out.read_text() == ""


class TestMaskToYoloBox:
    """Amodal axis-aligned bbox з бінарної маски (повний силует)."""

    def test_basic(self):
        m = np.zeros((1080, 1920), np.uint8)
        m[500:540, 900:980] = 255   # 80 wide × 40 high
        b = mask_to_yolo_box(m, 7, min_side_px=6)
        assert b.cls == 7
        assert b.w == pytest.approx(80 / 1920)
        assert b.h == pytest.approx(40 / 1080)
        assert b.xc == pytest.approx((900 + 80 / 2) / 1920)

    def test_empty_mask_none(self):
        assert mask_to_yolo_box(np.zeros((100, 100), np.uint8), 7, 6) is None

    def test_subthreshold_none(self):
        m = np.zeros((100, 100), np.uint8)
        m[10:14, 10:13] = 255   # 3×4 < 6
        assert mask_to_yolo_box(m, 7, 6) is None

    def test_amodal_full_silhouette_larger_than_visible(self):
        # amodal (full) даватиме більший bbox ніж visible (occluded) — стабільність
        full = np.zeros((200, 200), np.uint8)
        full[50:100, 50:120] = 255
        visible = full.copy()
        visible[50:100, 90:120] = 0     # правий бік «перекрито»
        bf = mask_to_yolo_box(full, 7, 6)
        bv = mask_to_yolo_box(visible, 7, 6)
        assert bf.w > bv.w              # amodal ширший — bbox не «стрибає»


class TestMaskToYoloObb:
    def test_axis_aligned_rect(self):
        m = np.zeros((200, 200), np.uint8)
        m[50:100, 40:160] = 255
        o = mask_to_yolo_obb(m, 1, min_side_px=6)
        assert isinstance(o, YoloObb)
        assert len(o.corners) == 4
        for x, y in o.corners:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    def test_empty_none(self):
        assert mask_to_yolo_obb(np.zeros((50, 50), np.uint8), 1, 6) is None

    def test_line_format_8_coords(self):
        m = np.zeros((100, 100), np.uint8)
        m[20:60, 20:80] = 255
        o = mask_to_yolo_obb(m, 3, 6)
        parts = o.to_line().split()
        assert parts[0] == "3"
        assert len(parts) == 9   # cls + 4 corners × 2

    def test_write_obb_roundtrip(self, tmp_path):
        m = np.zeros((100, 100), np.uint8)
        m[20:60, 20:80] = 255
        o = mask_to_yolo_obb(m, 3, 6)
        out = tmp_path / "o.txt"
        write_yolo_obb([o], out)
        assert out.read_text().strip().startswith("3 ")

    def test_write_obb_empty(self, tmp_path):
        out = tmp_path / "o.txt"
        write_yolo_obb([], out)
        assert out.read_text() == ""
