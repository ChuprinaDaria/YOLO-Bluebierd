"""Експорт layout відкритої/збереженої .blend сцени у YAML.

Зчитує позиції/rotation/scale всіх не-template/не-camera/не-light obj та групує за
prefix-категорією (Vehicle_, tree_g_, crater_, destroyed_, ruin_, mortar_, landmark_).
Lifespan: швидкий dump для подальшого reproducibility або refeed у render pipeline.

Запуск:
    blender --background --factory-startup PATH_TO.blend \
        --python scripts/export_scene_layout.py -- --output PATH_TO.yaml
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils

CATEGORY_PREFIXES = {
    "vehicle": ("Vehicle_",),
    "tree_green": ("tree_g_",),
    "tree_skeleton": ("tree_sk_",),
    "crater": ("crater_",),
    "destroyed": ("destroyed_",),
    "ruin": ("ruin_",),
    "mortar": ("mortar_",),
    "landmark": ("landmark_",),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(argv)


def _yaml_quote(s: str) -> str:
    return s.replace("'", "''")


def _categorize(name: str) -> str | None:
    for cat, prefixes in CATEGORY_PREFIXES.items():
        if any(name.startswith(p) for p in prefixes):
            return cat
    return None


def _is_template(obj) -> bool:
    return obj.hide_render and obj.hide_viewport and obj.location.z < -1000


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    grouped = {cat: [] for cat in CATEGORY_PREFIXES}
    skipped = []

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if _is_template(obj):
            skipped.append((obj.name, "hidden template"))
            continue
        cat = _categorize(obj.name)
        if cat is None:
            skipped.append((obj.name, "no category prefix"))
            continue

        loc = obj.location
        if obj.rotation_mode == 'QUATERNION':
            euler = obj.rotation_quaternion.to_euler('XYZ')
        else:
            euler = obj.rotation_euler

        grouped[cat].append({
            "name": obj.name,
            "location": [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)],
            "rotation_euler_deg": [
                round(math.degrees(euler.x), 2),
                round(math.degrees(euler.y), 2),
                round(math.degrees(euler.z), 2),
            ],
            "scale": [round(obj.scale.x, 4), round(obj.scale.y, 4), round(obj.scale.z, 4)],
        })

    # Сцена metadata
    scene_meta = {
        "blend_file": str(bpy.data.filepath) if bpy.data.filepath else "<unsaved>",
        "frame": bpy.context.scene.frame_current,
        "render_engine": bpy.context.scene.render.engine,
    }
    cam = bpy.context.scene.camera
    if cam:
        scene_meta["camera"] = {
            "name": cam.name,
            "location": [round(cam.location.x, 3), round(cam.location.y, 3), round(cam.location.z, 3)],
            "rotation_euler_deg": [
                round(math.degrees(cam.rotation_euler.x), 2),
                round(math.degrees(cam.rotation_euler.y), 2),
                round(math.degrees(cam.rotation_euler.z), 2),
            ],
            "lens_mm": round(cam.data.lens, 2),
            "sensor_width_mm": round(cam.data.sensor_width, 2),
        }

    # Manual YAML (без PyYAML — Blender stock не має)
    lines = []
    lines.append(f"# Layout export from {scene_meta['blend_file']}")
    lines.append("scene:")
    lines.append(f"  blend_file: '{_yaml_quote(scene_meta['blend_file'])}'")
    lines.append(f"  frame: {scene_meta['frame']}")
    lines.append(f"  render_engine: '{scene_meta['render_engine']}'")
    if "camera" in scene_meta:
        c = scene_meta["camera"]
        lines.append("  camera:")
        lines.append(f"    name: '{c['name']}'")
        lines.append(f"    location: {c['location']}")
        lines.append(f"    rotation_euler_deg: {c['rotation_euler_deg']}")
        lines.append(f"    lens_mm: {c['lens_mm']}")
        lines.append(f"    sensor_width_mm: {c['sensor_width_mm']}")
    lines.append("")
    lines.append("objects:")
    for cat in sorted(grouped):
        items = grouped[cat]
        if not items:
            continue
        lines.append(f"  {cat}:  # {len(items)} items")
        for it in items:
            lines.append(f"    - name: '{it['name']}'")
            lines.append(f"      location: {it['location']}")
            lines.append(f"      rotation_euler_deg: {it['rotation_euler_deg']}")
            lines.append(f"      scale: {it['scale']}")
    if skipped:
        lines.append("")
        lines.append("# Skipped (не лягло у жодну категорію):")
        for name, why in skipped[:20]:
            lines.append(f"#   {name}  ({why})")
        if len(skipped) > 20:
            lines.append(f"#   ... ще {len(skipped) - 20}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output}")
    print(f"[stats] vehicles={len(grouped['vehicle'])} "
          f"trees_g={len(grouped['tree_green'])} "
          f"craters={len(grouped['crater'])} "
          f"destroyed={len(grouped['destroyed'])} "
          f"ruins={len(grouped['ruin'])} "
          f"mortars={len(grouped['mortar'])} "
          f"landmarks={len(grouped['landmark'])}")


if __name__ == "__main__":
    main()
