"""Unit tests для datasetforge.engine.bbox_extractor (pure-Python частина).

Ключова поведінка: coco_to_yolo звітує n_dropped — кількість bbox, викинутих
min-side фільтром. render_runner на основі цього ВІДКИДАЄ кадр цілком
(видима-але-нерозмічена техніка = отруйний false negative).
"""

from __future__ import annotations

import json

import pytest

from datasetforge.engine.bbox_extractor import YoloBox, coco_to_yolo, write_yolo_label


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
