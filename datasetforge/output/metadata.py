"""Writer для metadata sidecar JSON поряд з кадром.

Кожен кадр `<stem>.jpg` мусить мати `<stem>.json` з усіма параметрами генерації
для downstream аналізу (per-altitude mAP, per-season etc).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FrameMetadata:
    image_id: str
    source: str  # "synthetic" | "real_osint" | "real_brave1"
    dataset_version: str  # df-v1.2.0
    seed: int

    altitude_m: float
    view_angle_deg: float
    hfov_deg: float
    sensor_res: tuple[int, int]
    modality: str  # EO | IR

    season: str
    landscape: str
    weather: str
    time_of_day: str

    degradation: dict  # jpeg_q, blur_px, atmo_strength, ...
    class_name: str
    has_targets: bool  # False -> hard negative

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
