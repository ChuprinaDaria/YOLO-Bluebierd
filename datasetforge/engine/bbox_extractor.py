"""COCO → YOLO bbox conversion + min-side filter.

BlenderProc writes COCO JSON; ми конвертуємо у YOLO TXT і фільтруємо bbox < min_side_px.

Pure-Python функції — без bpy/bproc — тестуються локально.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class YoloObb:
    """Oriented bbox — YOLO OBB формат: cls x1 y1 x2 y2 x3 y3 x4 y4 (norm 0-1).

    Для oblique-ракурсу і витягнутої техніки axis-aligned bbox тягне багато фону
    (ствол/корпус під кутом). OBB щільно обгортає силует → кращий IoU і
    орієнтація (ствол vs корпус) помічніша для розрізнення танк/БТР.
    """
    cls: int
    corners: tuple[tuple[float, float], float, tuple[float, float], float,
                   tuple[float, float], float, tuple[float, float], float]

    def to_line(self) -> str:
        pts = " ".join(f"{v:.6f}" for xy in self.corners for v in xy)
        return f"{self.cls} {pts}"


def mask_to_yolo_box(
    mask: "object",  # np.ndarray HxW, >0 = target
    class_id: int,
    min_side_px: int = 6,
) -> YoloBox | None:
    """Axis-aligned bbox з бінарної маски (amodal, коли маска = повний силует).

    Повертає None якщо маска порожня або min-side < порога. Numpy lazy-imported
    щоб pure-Python частина модуля лишалась importable без numpy.
    """
    import numpy as np

    m = np.asarray(mask)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w_px, h_px = (x1 - x0 + 1), (y1 - y0 + 1)
    if min(w_px, h_px) < min_side_px:
        return None
    H, W = m.shape[:2]
    return YoloBox(
        cls=class_id,
        xc=(x0 + w_px / 2.0) / W,
        yc=(y0 + h_px / 2.0) / H,
        w=w_px / W,
        h=h_px / H,
    )


def mask_to_yolo_obb(
    mask: "object",  # np.ndarray HxW, >0 = target
    class_id: int,
    min_side_px: int = 6,
) -> YoloObb | None:
    """Oriented bbox з бінарної маски через cv2.minAreaRect (min-area rectangle).

    Повертає None якщо маска порожня / min-side < порога. Кути нормовані 0-1,
    порядок — як повертає cv2.boxPoints (за годинниковою від низу).
    """
    import cv2
    import numpy as np

    m = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    if min(rw, rh) < min_side_px:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(pts))  # 4×2
    H, W = m.shape[:2]
    corners = tuple((float(x) / W, float(y) / H) for x, y in box)
    return YoloObb(cls=class_id, corners=corners)


def write_yolo_obb(boxes: list[YoloObb], path: Path) -> None:
    path.write_text("\n".join(b.to_line() for b in boxes) + ("\n" if boxes else ""),
                    encoding="utf-8")


def coco_xywh_to_yolo(
    coco_box: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    """COCO (xmin, ymin, w, h) у pixels → YOLO (xc, yc, w, h) normalized 0..1."""
    x, y, w, h = coco_box
    return (
        (x + w / 2) / image_w,
        (y + h / 2) / image_h,
        w / image_w,
        h / image_h,
    )


def coco_to_yolo(
    coco_json_path: Path,
    image_w: int,
    image_h: int,
    min_side_px: int = 6,
) -> list[tuple[str, list[YoloBox], int]]:
    """Читає BlenderProc-written COCO JSON.

    Повертає [(image_filename, [YoloBox, ...], n_dropped), ...].

    n_dropped = кількість bbox, викинутих min-side фільтром. КРИТИЧНО: кадр,
    де техніка видима, але всі bbox відфільтровані, — це НЕ hard negative,
    а отруєний false-negative (модель вчиться ігнорувати техніку). Рішення
    що робити з таким кадром (discard) приймає caller — render_runner.
    """
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))
    img_by_id = {img["id"]: img for img in data["images"]}
    boxes_by_img: dict[int, list[YoloBox]] = {iid: [] for iid in img_by_id}
    dropped_by_img: dict[int, int] = {iid: 0 for iid in img_by_id}
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        if min(w, h) < min_side_px:
            dropped_by_img[ann["image_id"]] += 1
            continue
        img = img_by_id[ann["image_id"]]
        xc, yc, nw, nh = coco_xywh_to_yolo((x, y, w, h), img["width"], img["height"])
        boxes_by_img[ann["image_id"]].append(
            YoloBox(
                cls=ann["category_id"],
                xc=xc, yc=yc, w=nw, h=nh,
                is_truncated=bool(ann.get("iscrowd", 0)),
            )
        )
    return [
        (img_by_id[iid]["file_name"], boxes, dropped_by_img[iid])
        for iid, boxes in boxes_by_img.items()
    ]


def write_yolo_label(boxes: list[YoloBox], path: Path) -> None:
    path.write_text("\n".join(b.to_line() for b in boxes) + ("\n" if boxes else ""),
                    encoding="utf-8")


def extract_from_3d_object(
    obj_name: str,
    class_id: int,
    image_w: int,
    image_h: int,
    min_side_px: int = 6,
) -> YoloBox | None:
    """3D bbox projection fallback коли BlenderProc COCO не використовується.

    Lazy import bpy. Беремо 8 вершин bound_box, проєктуємо у camera space,
    обчислюємо axis-aligned 2D bbox. Якщо min-side < min_side_px → None.
    """
    import bpy
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    cam = scene.camera
    obj = bpy.data.objects[obj_name]
    coords_2d = []
    for v in obj.bound_box:
        world_co = obj.matrix_world @ __mathutils_vec(v)
        cam_co = world_to_camera_view(scene, cam, world_co)
        coords_2d.append((cam_co.x, 1 - cam_co.y))  # Blender y-up → image y-down
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]
    x_min, x_max = max(0, min(xs)), min(1, max(xs))
    y_min, y_max = max(0, min(ys)), min(1, max(ys))
    w, h = x_max - x_min, y_max - y_min
    if w * image_w < min_side_px or h * image_h < min_side_px:
        return None
    return YoloBox(
        cls=class_id,
        xc=x_min + w / 2,
        yc=y_min + h / 2,
        w=w,
        h=h,
    )


def __mathutils_vec(v):
    from mathutils import Vector
    return Vector(v)
