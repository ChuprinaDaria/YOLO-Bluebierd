"""Headless build .blend preview сцени: ландшафт + reference drone камера + опційні props.

Запуск (приклад):
    blender --background --factory-startup --python scripts/build_landscape_preview.py -- \
        --landscape datasetforge/assets/landscapes/sutton_hoo_*.glb \
        --season autumn_mud --add-vehicle --vehicle-type kamaz \
        --add-trees-green 8 --add-craters 6 --add-destroyed 1 \
        --output datasetforge/blender_scenes/v8.blend

Камера — reference drone з v1_light_vehicle.yaml (hfov=15°, distance=300м, view_angle=20°).
Surface-normal alignment: vehicle/props orient'ються до нормалі ландшафту під ними.
Template-instance pattern: pack-glb імпортується один раз, scatter через linked-copy.
"""
import argparse
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils

REPO = Path("/home/dchuprina/YOLO -Bluebierd")
ASSETS = REPO / "datasetforge/assets"

HDRI_BY_SEASON = {
    "summer": ASSETS / "hdri/summer/kloofendal_43d_clear_puresky_2k.hdr",
    "autumn_mud": ASSETS / "hdri/autumn_mud/kloppenheim_06_puresky_2k.hdr",
    "winter": next((ASSETS / "hdri/winter").glob("*.hdr"), None) if (ASSETS / "hdri/winter").exists() else None,
    "spring": next((ASSETS / "hdri/spring").glob("*.hdr"), None) if (ASSETS / "hdri/spring").exists() else None,
}

VEHICLES = {
    "tigr": (ASSETS / "models/light_vehicle/gaz-2330_tigr.glb", 5.0, "Vehicle_GAZ_Tigr"),
    "kamaz": (ASSETS / "models/truck_logistics/russian_kamaz-5350_rosgvardiya.glb", 8.0, "Vehicle_Kamaz"),
    "tank": (ASSETS / "models/tank/russian_t-72b3.glb", 10.0, "Vehicle_T72"),
}
DESTROYED_TANK = ASSETS / "models/destroyed_proc/destroyed_russian_tank_t-72b.glb"

RUINS = [
    ASSETS / "props/buildings/damaged_school_building.glb",
    ASSETS / "props/buildings/school_in_myrne_mykolaiv_oblast_ukraine.glb",
]
# trees_low_poly.glb видалений з pool: поламаний asset (579×1100м bbox, монолітні mesh).
TREE_PACK = ASSETS / "props/vegetation/low_poly_forest_tree_pack.glb"
TREE_SKELETONS = [
    ASSETS / "props/vegetation/tree_free_extra_details_scan.glb",
    ASSETS / "props/vegetation/trees.glb",
]
CRATERS = ASSETS / "props/effects/bomb_craters_pack.glb"
MORTAR_STRIKES = ASSETS / "props/effects/mortar_strike.glb"
FAR_LANDMARKS = [
    ASSETS / "props/buildings/donetsk_intl_airport_terminal.glb",
    ASSETS / "props/buildings/donetsk_intl_airport_atc.glb",
    ASSETS / "props/misc/railway_tank.glb",
]

# Lazy-init template caches: pack імпортується один раз, scatter інстансує linked-copy.
_CRATER_TEMPLATES: list | None = None
_TREE_TEMPLATES: list | None = None


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--landscape", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--season", default="summer", choices=["summer", "autumn_mud", "winter", "spring"])
    p.add_argument("--add-vehicle", action="store_true")
    p.add_argument("--vehicle-type", default="tigr", choices=list(VEHICLES.keys()))
    p.add_argument("--add-ruins", type=int, default=0)
    p.add_argument("--add-trees-green", type=int, default=0)
    p.add_argument("--add-tree-skeletons", type=int, default=0)
    p.add_argument("--add-destroyed", type=int, default=0)
    p.add_argument("--add-craters", type=int, default=0)
    p.add_argument("--add-mortar-strikes", type=int, default=0)
    p.add_argument("--add-far-landmarks", type=int, default=0)
    p.add_argument("--add-fog", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pivot-x", type=float, default=None,
                   help="X координата центру сцени (default = центр ландшафту)")
    p.add_argument("--pivot-y", type=float, default=None,
                   help="Y координата центру сцени (default = центр ландшафту)")
    return p.parse_args(argv)


def cleanup():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def _bbox_world(objs):
    pts = []
    for o in objs:
        for v in o.bound_box:
            pts.append(o.matrix_world @ mathutils.Vector(v))
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def import_glb(path: Path, normalize_max_dim: float | None = None,
               lift_to_ground: bool = True, place_at_xyz=None,
               join_meshes: bool = False, name_hint: str | None = None,
               drop_flat_z_below: float | None = None):
    """Імпорт .glb з parent_clear+transform_apply (фікси scene_builder.py).
    join_meshes=True — з'єднати всі mesh частини у один Object (щоб G рухав цілісно).
    drop_flat_z_below — фільтр flat decals (наприклад сіра підкладка у crater pack).
    Повертає список mesh objects (1 елемент якщо join_meshes=True)."""
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(path))
    new_names = list(set(bpy.data.objects.keys()) - before)
    new_objs = [bpy.data.objects[n] for n in new_names]

    bpy.ops.object.select_all(action='DESELECT')
    for o in new_objs:
        o.select_set(True)
    if new_objs:
        bpy.context.view_layer.objects.active = new_objs[0]
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.select_all(action='DESELECT')
        meshes = [o for o in new_objs if o.type == "MESH"]
        for m in meshes:
            m.select_set(True)
        if meshes:
            bpy.context.view_layer.objects.active = meshes[0]
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')

    for n in list(new_names):
        o = bpy.data.objects.get(n)
        if o and o.type == "EMPTY":
            bpy.data.objects.remove(o, do_unlink=True)

    meshes = [bpy.data.objects[n] for n in new_names
              if bpy.data.objects.get(n) and bpy.data.objects[n].type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no MESH in {path}")

    if drop_flat_z_below is not None:
        keep = []
        for m in meshes:
            pts = [m.matrix_world @ mathutils.Vector(v) for v in m.bound_box]
            zs = [p.z for p in pts]
            if (max(zs) - min(zs)) >= drop_flat_z_below:
                keep.append(m)
            else:
                bpy.data.objects.remove(m, do_unlink=True)
        meshes = keep
        if not meshes:
            raise RuntimeError(f"all meshes dropped by drop_flat_z_below in {path}")

    if normalize_max_dim:
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _bbox_world(meshes)
        current_max = max(xmx - xmn, ymx - ymn, zmx - zmn)
        if current_max > 0.01:
            sf = normalize_max_dim / current_max
            if not (0.95 < sf < 1.05):
                for m in meshes:
                    m.scale = tuple(s * sf for s in m.scale)
                    m.location = tuple(l * sf for l in m.location)
                bpy.context.view_layer.update()

    if lift_to_ground:
        (_, _, zmn), _ = _bbox_world(meshes)
        if abs(zmn) > 0.001:
            for m in meshes:
                m.location.z -= zmn
            bpy.context.view_layer.update()

    if place_at_xyz is not None:
        (xmn, ymn, _), (xmx, ymx, _) = _bbox_world(meshes)
        cx = (xmn + xmx) / 2
        cy = (ymn + ymx) / 2
        dx = place_at_xyz[0] - cx
        dy = place_at_xyz[1] - cy
        dz = place_at_xyz[2]
        for m in meshes:
            m.location.x += dx
            m.location.y += dy
            m.location.z += dz
        bpy.context.view_layer.update()

    if join_meshes and len(meshes) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        joined = bpy.context.view_layer.objects.active
        if name_hint:
            joined.name = name_hint
        meshes = [joined]
        bpy.context.view_layer.update()

    return meshes


def raycast_surface(x: float, y: float, fallback_z: float = 0.0):
    """Раз з неба у точці (x,y). Повертає (z, normal). На miss — (fallback_z, world +Z)."""
    origin = mathutils.Vector((x, y, 10000.0))
    direction = mathutils.Vector((0, 0, -1))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result, location, normal, *_ = bpy.context.scene.ray_cast(depsgraph, origin, direction)
    if result and normal.length > 0.01:
        return location.z, normal
    return fallback_z, mathutils.Vector((0, 0, 1))


def align_to_normal(obj, normal: mathutils.Vector, yaw: float):
    """Поставити local +Z обʼєкта вздовж surface normal + yaw spin навколо нової up-осі.
    На майже-flat (normal.z > 0.98) — лишити pure yaw euler, уникає floating-point jitter."""
    if normal.length < 0.01:
        normal = mathutils.Vector((0, 0, 1))
    if normal.z > 0.98:
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, yaw)
        return
    align_quat = normal.to_track_quat('Z', 'Y')
    yaw_quat = mathutils.Quaternion((0, 0, 1), yaw)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = yaw_quat @ align_quat


def _park_hidden(obj):
    """Спрятати template-обʼєкт під сценою щоб не лишатись у viewport/render."""
    obj.location = (0.0, 0.0, -10000.0)
    obj.hide_viewport = True
    obj.hide_render = True


def _import_crater_templates() -> list:
    """Імпорт bomb_craters_pack як hidden templates: 23 окремі кратери, scale up до 1.5-3м."""
    meshes = import_glb(CRATERS, normalize_max_dim=None, lift_to_ground=False,
                        join_meshes=False, drop_flat_z_below=0.05)
    rng = random.Random(0)
    for m in meshes:
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _bbox_world([m])
        diameter = max(xmx - xmn, ymx - ymn)
        if diameter < 0.01:
            continue
        target = rng.uniform(1.5, 3.0)
        sf = target / diameter
        m.scale = tuple(s * sf for s in m.scale)
    bpy.context.view_layer.update()
    for m in meshes:
        _park_hidden(m)
    bpy.context.view_layer.update()
    print(f"[crater templates] {len(meshes)} ready")
    return meshes


def _import_tree_templates() -> list:
    """Імпорт low_poly_forest_tree_pack як hidden templates.

    Pack містить розрізнені компоненти: Tree_Trunk_* (стовбур без крони),
    Tree_Branches_* (крона без стовбура), Background_Tree_Atlas_* (повне дерево
    як crossed-planes billboard), Rocks_*, Cube. Беремо лише Background_Tree_Atlas
    бо вони цілісні; окремі trunk/branches без напарника виглядають як обрубки.
    Нормалізуємо до 8м (дуб/тополя на UA-території).
    """
    meshes = import_glb(TREE_PACK, normalize_max_dim=None, lift_to_ground=False,
                        join_meshes=False)
    target_h = 8.0
    keep = []
    for m in meshes:
        if not m.name.startswith("Background_Tree_Atlas"):
            bpy.data.objects.remove(m, do_unlink=True)
            continue
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _bbox_world([m])
        h = zmx - zmn
        if h < 8.0:
            bpy.data.objects.remove(m, do_unlink=True)
            continue
        sf = target_h / h
        m.scale = tuple(s * sf for s in m.scale)
        keep.append(m)
    bpy.context.view_layer.update()
    for m in keep:
        _park_hidden(m)
    bpy.context.view_layer.update()
    print(f"[tree templates] {len(keep)} Background_Tree_Atlas ready (pool {len(meshes)})")
    return keep


def _instance_from_template(template, xyz, yaw: float, normal: mathutils.Vector,
                            extra_scale: float = 1.0, name: str | None = None):
    """Linked-copy template'у у точку xyz з surface-aligned rotation.
    Mesh data shared (template.data) — економія пам'яті для preview."""
    new = template.copy()
    new.data = template.data
    bpy.context.collection.objects.link(new)
    new.hide_viewport = False
    new.hide_render = False
    new.location = mathutils.Vector(xyz)
    if abs(extra_scale - 1.0) > 0.001:
        new.scale = tuple(s * extra_scale for s in template.scale)
    else:
        new.scale = template.scale
    align_to_normal(new, normal, yaw)
    if name:
        new.name = name
    return new


def _scatter_craters(n: int, around_xy, seed: int):
    """Per-1 crater scatter через template-instance pattern."""
    global _CRATER_TEMPLATES
    if _CRATER_TEMPLATES is None:
        _CRATER_TEMPLATES = _import_crater_templates()
    rng = random.Random(seed)
    for i in range(n):
        r = rng.uniform(10, 80)
        theta = rng.uniform(0, 2 * math.pi)
        x = around_xy[0] + r * math.cos(theta)
        y = around_xy[1] + r * math.sin(theta)
        z_hit, normal = raycast_surface(x, y, fallback_z=0.0)
        src = rng.choice(_CRATER_TEMPLATES)
        yaw = rng.uniform(0, 2 * math.pi)
        _instance_from_template(src, (x, y, z_hit), yaw, normal,
                                name=f"crater_{i+1}")
        bpy.context.view_layer.update()
        print(f"[crater {i+1}/{n}] @ ({x:.1f},{y:.1f},{z_hit:.1f}) n.z={normal.z:.2f}")


def _scatter_trees(n: int, around_xy, seed: int, label: str = "tree_g"):
    """Per-1 tree scatter через template-instance pattern з random scale variation."""
    global _TREE_TEMPLATES
    if _TREE_TEMPLATES is None:
        _TREE_TEMPLATES = _import_tree_templates()
    if not _TREE_TEMPLATES:
        print(f"[{label}] no tree templates, skip scatter")
        return
    rng = random.Random(seed)
    for i in range(n):
        r = rng.uniform(20, 100)
        theta = rng.uniform(0, 2 * math.pi)
        x = around_xy[0] + r * math.cos(theta)
        y = around_xy[1] + r * math.sin(theta)
        z_hit, normal = raycast_surface(x, y, fallback_z=0.0)
        src = rng.choice(_TREE_TEMPLATES)
        yaw = rng.uniform(0, 2 * math.pi)
        extra = rng.uniform(0.85, 1.15)
        _instance_from_template(src, (x, y, z_hit), yaw, normal,
                                extra_scale=extra, name=f"{label}_{i+1}")
        bpy.context.view_layer.update()
        print(f"[{label} {i+1}/{n}] @ ({x:.1f},{y:.1f},{z_hit:.1f}) scale=×{extra:.2f}")


def scatter_props(prop_glbs, n: int, around_xy, radius_min: float, radius_max: float,
                  normalize_max_dim: float, seed: int, label: str):
    """Generic scatter для standalone props (не з template caching).
    Join у один Object + surface-normal alignment."""
    rng = random.Random(seed)
    for i in range(n):
        glb = rng.choice(prop_glbs) if isinstance(prop_glbs, list) else prop_glbs
        r = rng.uniform(radius_min, radius_max)
        theta = rng.uniform(0, 2 * math.pi)
        x = around_xy[0] + r * math.cos(theta)
        y = around_xy[1] + r * math.sin(theta)
        z_hit, normal = raycast_surface(x, y, fallback_z=0.0)
        meshes = import_glb(glb, normalize_max_dim=normalize_max_dim,
                            lift_to_ground=True, place_at_xyz=(x, y, z_hit),
                            join_meshes=True, name_hint=f"{label}_{i+1}")
        yaw = rng.uniform(0, 2 * math.pi)
        for m in meshes:
            align_to_normal(m, normal, yaw)
        bpy.context.view_layer.update()
        print(f"[{label} {i+1}/{n}] {glb.name} @ ({x:.1f},{y:.1f},{z_hit:.1f}) n.z={normal.z:.2f}")


def setup_hdri(path: Path, strength: float = 1.0):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if path and path.exists():
        env.image = bpy.data.images.load(str(path))
    bg.inputs["Strength"].default_value = strength
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_fog_volume(density: float = 0.005):
    world = bpy.context.scene.world
    nt = world.node_tree
    vs = nt.nodes.new("ShaderNodeVolumeScatter")
    vs.inputs["Density"].default_value = density
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    nt.links.new(vs.outputs["Volume"], out.inputs["Volume"])
    bpy.context.scene.eevee.use_volumetric_lights = True
    bpy.context.scene.eevee.volumetric_end = 1000


def add_sun(energy: float = 3.0, zenith_deg: float = 35.0, azim_deg: float = 45.0):
    d = bpy.data.lights.new("Sun", type='SUN')
    d.energy = energy
    o = bpy.data.objects.new("Sun", d)
    bpy.context.collection.objects.link(o)
    o.rotation_euler = (math.radians(zenith_deg), 0, math.radians(azim_deg))


def add_drone_camera(target, distance_m: float = 300.0, view_angle_deg: float = 20.0,
                     hfov_deg: float = 15.0, azim_deg: float = 30.0):
    elev = math.radians(view_angle_deg)
    azim = math.radians(azim_deg)
    h = distance_m * math.sin(elev)
    xy = distance_m * math.cos(elev)
    cx = target[0] + xy * math.cos(azim)
    cy = target[1] + xy * math.sin(azim)
    cz = target[2] + h

    cd = bpy.data.cameras.new("DroneCam")
    co = bpy.data.objects.new("DroneCam", cd)
    bpy.context.collection.objects.link(co)
    co.location = (cx, cy, cz)
    direction = mathutils.Vector(target) - mathutils.Vector(co.location)
    co.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    sensor_w = 35.0
    cd.lens = sensor_w / (2 * math.tan(math.radians(hfov_deg) / 2))
    cd.sensor_width = sensor_w
    cd.clip_end = max(distance_m * 5, 20000)

    bpy.context.scene.camera = co
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 640
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    return co


def setup_viewport(clip_end: float = 20000.0):
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                space.clip_end = clip_end
                space.clip_start = 0.1
                space.shading.type = 'MATERIAL'
                if space.region_3d:
                    space.region_3d.view_perspective = 'CAMERA'


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cleanup()

    print(f"[landscape] importing {args.landscape.name}")
    landscape = import_glb(args.landscape, normalize_max_dim=None, lift_to_ground=True)
    (lxmn, lymn, lzmn), (lxmx, lymx, lzmx) = _bbox_world(landscape)
    landscape_center_xy = ((lxmn + lxmx) / 2, (lymn + lymx) / 2)
    landscape_extent = min(lxmx - lxmn, lymx - lymn)
    print(f"[landscape] extent={landscape_extent:.1f}m center_xy={landscape_center_xy}")

    hdri = HDRI_BY_SEASON.get(args.season)
    setup_hdri(hdri, strength=1.0)
    add_sun(energy=3.0 if args.season != "autumn_mud" else 2.0)

    if args.pivot_x is not None and args.pivot_y is not None:
        vehicle_xy = (args.pivot_x, args.pivot_y)
        print(f"[pivot] custom @ {vehicle_xy}")
    else:
        vehicle_xy = landscape_center_xy
    vehicle_z, vehicle_normal = raycast_surface(vehicle_xy[0], vehicle_xy[1], fallback_z=0.0)

    if args.add_vehicle:
        v_path, v_size, v_name = VEHICLES[args.vehicle_type]
        print(f"[vehicle] {args.vehicle_type} @ {vehicle_xy} z={vehicle_z:.1f} n.z={vehicle_normal.z:.2f}")
        meshes = import_glb(v_path, normalize_max_dim=v_size, lift_to_ground=True,
                            place_at_xyz=(vehicle_xy[0], vehicle_xy[1], vehicle_z),
                            join_meshes=True, name_hint=v_name)
        for m in meshes:
            align_to_normal(m, vehicle_normal, yaw=0.0)
        bpy.context.view_layer.update()

    if args.add_ruins:
        scatter_props(RUINS, args.add_ruins, vehicle_xy, 30, 80,
                      normalize_max_dim=20.0, seed=args.seed + 10, label="ruin")
    if args.add_trees_green:
        _scatter_trees(args.add_trees_green, vehicle_xy, args.seed + 20, label="tree_g")
    if args.add_tree_skeletons:
        scatter_props(TREE_SKELETONS, args.add_tree_skeletons, vehicle_xy, 20, 100,
                      normalize_max_dim=15.0, seed=args.seed + 30, label="tree_sk")
    if args.add_destroyed:
        scatter_props([DESTROYED_TANK], args.add_destroyed, vehicle_xy, 15, 50,
                      normalize_max_dim=7.0, seed=args.seed + 40, label="destroyed")
    if args.add_craters:
        _scatter_craters(args.add_craters, vehicle_xy, args.seed + 50)
    if args.add_mortar_strikes:
        scatter_props([MORTAR_STRIKES], args.add_mortar_strikes, vehicle_xy, 15, 70,
                      normalize_max_dim=8.0, seed=args.seed + 60, label="mortar")
    if args.add_far_landmarks:
        scatter_props(FAR_LANDMARKS, args.add_far_landmarks, vehicle_xy, 150, 350,
                      normalize_max_dim=40.0, seed=args.seed + 70, label="landmark")

    target = (vehicle_xy[0], vehicle_xy[1], vehicle_z + 1.5)
    add_drone_camera(target, distance_m=300, view_angle_deg=20, hfov_deg=15)

    if args.add_fog:
        add_fog_volume(density=0.003)

    setup_viewport()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
