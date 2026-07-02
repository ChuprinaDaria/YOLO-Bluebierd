"""COCO → YOLO + YoloBox.to_line + min_side_px filter."""

import json
import math
from pathlib import Path

from datasetforge.engine.bbox_extractor import (
    YoloBox, coco_to_yolo, coco_xywh_to_yolo, write_yolo_label,
)


def test_yolo_box_to_line_format():
    b = YoloBox(cls=7, xc=0.5, yc=0.5, w=0.1, h=0.2)
    line = b.to_line()
    parts = line.split()
    assert parts[0] == "7"
    assert all(len(p.split(".")[1]) == 6 for p in parts[1:])


def test_coco_xywh_to_yolo_normalization():
    # COCO (10, 20, 100, 200) у 640×640 → центр (60, 120), w/h 100/200
    xc, yc, w, h = coco_xywh_to_yolo((10, 20, 100, 200), 640, 640)
    assert math.isclose(xc, 60 / 640)
    assert math.isclose(yc, 120 / 640)
    assert math.isclose(w, 100 / 640)
    assert math.isclose(h, 200 / 640)


def test_coco_to_yolo_filters_small(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "f.jpg", "width": 640, "height": 640}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [100, 100, 50, 50]},
            {"image_id": 1, "category_id": 7, "bbox": [200, 200, 5, 5]},  # min_side < 10
        ],
        "categories": [{"id": 7, "name": "light_vehicle"}],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(coco))
    result = coco_to_yolo(p, image_w=640, image_h=640, min_side_px=10)
    assert len(result) == 1
    fname, boxes = result[0]
    assert fname == "f.jpg"
    assert len(boxes) == 1
    assert boxes[0].cls == 7


def test_write_yolo_label_round_trip(tmp_path: Path):
    boxes = [YoloBox(cls=7, xc=0.5, yc=0.5, w=0.1, h=0.2),
             YoloBox(cls=0, xc=0.2, yc=0.3, w=0.05, h=0.08)]
    p = tmp_path / "label.txt"
    write_yolo_label(boxes, p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("7 ")
    assert lines[1].startswith("0 ")


def test_write_yolo_label_empty_is_valid_negative(tmp_path: Path):
    p = tmp_path / "neg.txt"
    write_yolo_label([], p)
    assert p.exists()
    assert p.read_text() == ""
