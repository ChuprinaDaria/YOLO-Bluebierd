"""Pre-import normalizer для .blend моделей з Sketchfab/CGTrader/etc.

Запускати ОДИН раз після завантаження кожної нової моделі:
    blender --background --python datasetforge/assets/normalize_blend.py -- \\
        --input raw_download.blend --output assets/models/<class>/<variant>.blend

Кроки:
  1. Scale check: якщо модель у cm (типово для Sketchfab) → scale 0.01 → m.
  2. Origin re-center: origin на bottom-center bbox (vehicle стоїть на землі).
  3. Texture re-link: знаходить missing image refs у тій же папці.
  4. Z-up axis check (Blender default — Z-up; Sketchfab експорт інколи Y-up).
  5. Apply transforms + freeze.

Запускається з bpy → НЕ importable локально без Blender.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-units", choices=["m", "cm", "mm", "auto"], default="auto",
                    help="Якщо auto — детектимо за bbox size (>20 = cm, >2000 = mm).")
    ap.add_argument("--y-up", action="store_true", help="Source Y-up, rotate to Z-up.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import bpy

    args = parse_args(argv if argv is not None else sys.argv[sys.argv.index("--") + 1:])

    # Завантажити вихідний .blend (або інший формат через відповідний importer)
    if args.input.suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(args.input))
    elif args.input.suffix.lower() in (".fbx",):
        bpy.ops.import_scene.fbx(filepath=str(args.input))
    elif args.input.suffix.lower() in (".obj",):
        bpy.ops.wm.obj_import(filepath=str(args.input))
    elif args.input.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(args.input))
    else:
        print(f"[err] unsupported format: {args.input.suffix}", file=sys.stderr)
        return 2

    # Cleanup: видалити default cube/camera/light
    for default_name in ("Cube", "Camera", "Light"):
        if default_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[default_name])

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("[err] no MESH after import", file=sys.stderr)
        return 2

    # 1. Y-up → Z-up
    if args.y_up:
        for o in meshes:
            o.rotation_euler[0] = 1.5708  # 90° X

    # 2. Scale auto-detect
    bpy.context.view_layer.update()
    bbox_size = _scene_bbox_size(meshes)
    max_dim = max(bbox_size)
    scale_factor = 1.0
    if args.source_units == "auto":
        if max_dim > 2000:
            scale_factor = 0.001  # mm → m
        elif max_dim > 20:
            scale_factor = 0.01  # cm → m
    elif args.source_units == "cm":
        scale_factor = 0.01
    elif args.source_units == "mm":
        scale_factor = 0.001

    if scale_factor != 1.0:
        for o in meshes:
            o.scale = (scale_factor, scale_factor, scale_factor)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 3. Origin → bottom-center
    bpy.context.view_layer.update()
    bbox = _scene_bbox(meshes)
    z_min = bbox[0][2]
    x_mid = (bbox[0][0] + bbox[1][0]) / 2
    y_mid = (bbox[0][1] + bbox[1][1]) / 2
    bpy.context.scene.cursor.location = (x_mid, y_mid, z_min)
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
        o.location = (0, 0, 0)

    # 4. Missing texture re-link (шукаємо у тій же папці що input)
    src_dir = args.input.parent
    for img in bpy.data.images:
        if img.filepath and not Path(img.filepath_from_user()).exists():
            cand = src_dir / Path(img.filepath).name
            if cand.exists():
                img.filepath = str(cand)
                img.reload()

    # 5. Зберегти у output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"[ok] normalized → {args.output}")
    return 0


def _scene_bbox(meshes):
    import mathutils
    coords = []
    for o in meshes:
        for v in o.bound_box:
            coords.append(o.matrix_world @ mathutils.Vector(v))
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    zs = [c.z for c in coords]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _scene_bbox_size(meshes):
    (x0, y0, z0), (x1, y1, z1) = _scene_bbox(meshes)
    return (x1 - x0, y1 - y0, z1 - z0)


if __name__ == "__main__":
    sys.exit(main())
