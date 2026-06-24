"""Скелет 2D YOLO bbox extractor з 3D bounding box.

Логіка:
1. Беремо 8 вершин 3D bbox мешу через `obj.bound_box`.
2. Проєктуємо на 2D через `bpy_extras.object_utils.world_to_camera_view`.
3. min/max по проєкціях → axis-aligned 2D bbox у normalized 0..1 YOLO координатах.
4. Перевіряємо чи частина обʼєкта видима (не повністю за кадром).
5. Якщо bbox <10 px (за pixel_budget threshold) — повертаємо None (тоді кадр може стати hard negative).

Імплементується під bpy; тут — інтерфейс.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YoloBox:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    is_truncated: bool = False
    is_occluded: bool = False

    def to_line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def extract_from_3d_object(  # noqa: ARG001
    obj_name: str,
    class_id: int,
    image_w: int,
    image_h: int,
    min_side_px: int = 10,
) -> YoloBox | None:
    """TODO: bpy implementation. Зараз — placeholder."""
    raise NotImplementedError("Implement under Blender Python (bpy)")
