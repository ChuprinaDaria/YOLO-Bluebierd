"""Blender scene builder через BlenderProc.

Запускається через `blenderproc run datasetforge/engine/render_runner.py ...`
(BlenderProc сам стартує Blender 4.x з потрібним bpy в process).

Importable локально без bpy — `bproc` import зроблений lazy всередині build_scene.
Це дозволяє юніт-тестам тестувати `CameraSpec`/`SceneRequest` дataclasses без Blender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraSpec:
    distance_m: float        # 3D line-of-sight camera → vehicle (200-1000 m)
    view_angle_deg: float    # елевація над горизонтом (10-30° drone cruising)
    hfov_deg: float
    sensor_width_mm: float = 35.0

    @property
    def focal_mm(self) -> float:
        return self.sensor_width_mm / (2 * math.tan(math.radians(self.hfov_deg) / 2))


@dataclass
class SceneRequest:
    class_name: str
    class_id: int
    model_path: Path
    hdri_path: Path
    ground_texture_path: Path
    camera: CameraSpec
    season: str
    landscape: str
    weather: str
    image_w: int
    image_h: int
    seed: int
    # Підкласти ґрунтову колію під техніку (для road-landscapes). Дає depth+RGB
    # реальну дорогу, яку Flux добудовує — фікс «техніка посеред поля».
    road_under_vehicle: bool = False


def build_scene(req: SceneRequest):
    """Будує одну сцену під рендер. Повертає (camera_pose_4x4, vehicle_mesh_objs, sun_info).

    sun_info — dict {sun_zenith_rad, sun_azim_rad, sun_elevation_deg, sun_azimuth_deg}.
    Споживає render_runner для metadata sidecar (downstream diffusion prompt
    "low south-west sun at 35 degrees elevation").

    Кроки:
      1. Завантажити vehicle (.blend / .glb / .gltf / .obj / .fbx).
      2. Tag category_id + normalize scale до 5м max-dim + random Z rotation.
      3. Ground plane 10×10км з seasonal PBR-текстурою.
      4. HDRI world background (strength=2.0) + explicit SUN light (energy=5).
      5. Camera pose: distance_m + view_angle_deg → H=d·sin(θ), xy=d·cos(θ).
      6. Camera intrinsics (focal_mm з hfov_deg, sensor_width).
    """
    # Lazy import: bproc + bpy доступні тільки коли запущено через `blenderproc run`.
    import bpy
    import blenderproc as bproc
    import numpy as np

    # 0. CLEANUP previous frame.
    # bproc.utility.reset_keyframes() у render_runner чистить ТІЛЬКИ анімаційні keyframes,
    # mesh+light об'єкти з попереднього кадру лишаються у сцені і накопичуються.
    # Без цього cleanup після 3-х кадрів у сцені 3 машини = "зліплені 3 моделі" артефакт.
    for obj in list(bpy.data.objects):
        if obj.type in ("MESH", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)
    # Orphan data cleanup щоб memory не роздувалась за 20+ frames (texture/mesh blocks).
    for collection in (bpy.data.meshes, bpy.data.materials,
                       bpy.data.images, bpy.data.lights):
        for item in list(collection):
            if not item.users:
                collection.remove(item)

    # 1. Завантажити vehicle (dispatch за extension)
    ext = req.model_path.suffix.lower()
    if ext == ".blend":
        objs = bproc.loader.load_blend(str(req.model_path))
    elif ext in (".glb", ".gltf", ".fbx"):
        before = set(bpy.data.objects.keys())
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(req.model_path))
        else:
            bpy.ops.import_scene.fbx(filepath=str(req.model_path))
        new_names = list(set(bpy.data.objects.keys()) - before)
        new_objs = [bpy.data.objects[n] for n in new_names]

        # glTF спека = Y-up; Blender = Z-up. Importer ставить корекційну ротацію
        # (Euler(π/2,0,0)) на ROOT EMPTY, mesh children мають identity local rotation.
        # Якщо ми потім robimо set_rotation_euler([0,0,z]) на mesh — z-axis у parent
        # frame ≠ world Z → vehicle качається догори ногами / на бік.
        # Fix: parent_clear (KEEP_TRANSFORM) + transform_apply бейкає parent matrix
        # у mesh data, після цього local == world rotation.
        bpy.ops.object.select_all(action='DESELECT')
        for obj in new_objs:
            obj.select_set(True)
        if new_objs:
            bpy.context.view_layer.objects.active = new_objs[0]
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            # Select лише mesh-and: transform_apply не любить EMPTY.
            bpy.ops.object.select_all(action='DESELECT')
            meshes_to_apply = [o for o in new_objs if o.type == "MESH"]
            for obj in meshes_to_apply:
                obj.select_set(True)
            if meshes_to_apply:
                bpy.context.view_layer.objects.active = meshes_to_apply[0]
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.select_all(action='DESELECT')

        # Прибрати глТФ root emptys — після parent_clear вони сироти і тільки шумлять
        # у segmentation map (хоча type=EMPTY → renderable=False, але data.images кеш росте).
        for n in list(new_names):
            obj = bpy.data.objects.get(n)
            if obj and obj.type == "EMPTY":
                bpy.data.objects.remove(obj, do_unlink=True)

        objs = [bproc.types.MeshObject(bpy.data.objects[n]) for n in new_names
                if bpy.data.objects.get(n) and bpy.data.objects[n].type == "MESH"]
    elif ext == ".obj":
        objs = bproc.loader.load_obj(str(req.model_path))
    else:
        raise RuntimeError(f"unsupported model format: {ext} ({req.model_path})")
    # objs з усіх loader-ів — вже MeshObject; додатковий filter не потрібен.
    vehicle_meshes = [o for o in objs if isinstance(o, bproc.types.MeshObject)]
    if not vehicle_meshes:
        raise RuntimeError(f"no MESH objects in {req.model_path}")
    for o in vehicle_meshes:
        o.set_cp("category_id", req.class_id)

    # Normalize vehicle scale: max-dim → ~5 м.
    # Захист від GLB-файлів з non-meters units (cm/dm/mm), що тягне камеру
    # всередину моделі і дає "50-100 см рендер" замість 200-1000 м.
    TARGET_MAX_DIM_M = 5.0
    combined_bbox = np.vstack([np.array(o.get_bound_box(local_coords=False))
                               for o in vehicle_meshes])
    dims = combined_bbox.max(axis=0) - combined_bbox.min(axis=0)
    current_max = float(dims.max())
    if current_max > 0.01:
        scale_factor = TARGET_MAX_DIM_M / current_max
        if not (0.95 < scale_factor < 1.05):
            for o in vehicle_meshes:
                cur_scale = o.get_scale()
                cur_loc = o.get_location()
                o.set_scale([cur_scale[i] * scale_factor for i in range(3)])
                o.set_location([cur_loc[i] * scale_factor for i in range(3)])
            bpy.context.view_layer.update()
            print(f"[scale] {req.model_path.name}: max_dim "
                  f"{current_max:.2f}m → {TARGET_MAX_DIM_M}m (×{scale_factor:.3f})")

    # Random Z rotation для variety. Після parent_clear+transform_apply вище local rotation
    # вже дорівнює world rotation, тому це чистий yaw навколо world Z (vehicle стоїть прямо).
    rng = np.random.default_rng(req.seed)
    z_rot = float(rng.uniform(0, 2 * math.pi))
    for o in vehicle_meshes:
        cur = o.get_rotation_euler()
        o.set_rotation_euler([cur[0], cur[1], cur[2] + z_rot])
    bpy.context.view_layer.update()

    # Lift vehicle щоб bbox bottom торкався ground plane (z=0).
    # Vehicle origin зазвичай у центрі моделі — без lift половина моделі тоне в землю.
    combined_bbox_post = np.vstack([np.array(o.get_bound_box(local_coords=False))
                                    for o in vehicle_meshes])
    min_z = float(combined_bbox_post[:, 2].min())
    if abs(min_z) > 0.001:
        for o in vehicle_meshes:
            cur_loc = o.get_location()
            o.set_location([cur_loc[0], cur_loc[1], cur_loc[2] - min_z])
        bpy.context.view_layer.update()

    # Центр vehicle після scale+rotation+lift (для камера-target).
    # Використовуємо combined bbox через ВСІ меш-частини, не першу — для multi-mesh
    # GLB (body + wheels окремо) перша мабуть body, її center зміщений.
    combined_bbox_final = np.vstack([np.array(o.get_bound_box(local_coords=False))
                                     for o in vehicle_meshes])
    center = (combined_bbox_final.max(axis=0) + combined_bbox_final.min(axis=0)) / 2
    dims_final = combined_bbox_final.max(axis=0) - combined_bbox_final.min(axis=0)
    # Sanity log: для car/light_vehicle height має бути ~0.4-0.6 від length.
    # Якщо height ≥ length → vehicle лежить на боку / догори ногами.
    h_to_l = dims_final[2] / max(dims_final[:2].max(), 0.01)
    print(f"[orient] {req.model_path.name}: dims=({dims_final[0]:.2f},{dims_final[1]:.2f},{dims_final[2]:.2f}) m "
          f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) h/L={h_to_l:.2f} "
          f"{'OK' if h_to_l < 0.8 else 'LIKELY-ON-SIDE'}")

    # 2. Ground plane 10km×10km щоб горизонт не вилазив (drone з 800м бачить ~15км до горизонту).
    ground = bproc.object.create_primitive("PLANE", scale=[5000, 5000, 1])
    ground.set_location([float(center[0]), float(center[1]), 0])
    ground.set_cp("category_id", 0)  # 0 = background, інакше segmentation падає
    # Apply seasonal ground texture (PBR диффузка з Poly Haven)
    if req.ground_texture_path.exists():
        gmat = bproc.material.create_material_from_texture(
            str(req.ground_texture_path), material_name=f"ground_{req.season}"
        )
        ground.replace_materials(gmat)

    # 2b. Road strip під технікою (опційно, для road-landscapes).
    # Вузька темніша ґрунтова смуга вздовж heading техніки (z_rot). Дає Flux у
    # depth+RGB реальну дорогу замість «техніка посеред поля». Guarded — будь-яка
    # помилка тут не валить рендер (road = nice-to-have, не критичний).
    ROAD_LANDSCAPES = ("dirt_road", "forest_belt")
    if req.road_under_vehicle and req.landscape in ROAD_LANDSCAPES:
        try:
            road_len_m = 70.0
            road_wid_m = 5.0
            road = bproc.object.create_primitive(
                "PLANE", scale=[road_len_m / 2.0, road_wid_m / 2.0, 1.0])
            # Трохи над ground (z=0) щоб не було z-fighting; на aerial масштабі непомітно.
            road.set_location([float(center[0]), float(center[1]), 0.03])
            road.set_rotation_euler([0.0, 0.0, z_rot])  # вздовж heading техніки
            road.set_cp("category_id", 0)               # background, не ламає segmentation
            road_mat = bproc.material.create("dirt_road")
            # Темно-коричнева утоптана земля, матова — контраст до трав'яного ground.
            road_mat.set_principled_shader_value("Base Color", [0.21, 0.16, 0.11, 1.0])
            road_mat.set_principled_shader_value("Roughness", 1.0)
            road_mat.set_principled_shader_value("Specular", 0.05)
            road.replace_materials(road_mat)
            print(f"[road] strip {road_len_m:.0f}×{road_wid_m:.0f}m під технікою "
                  f"(landscape={req.landscape})")
        except Exception as exc:
            print(f"[road] skip (non-fatal): {exc.__class__.__name__}: {exc}")

    # 3. World HDRI sky lighting (strength 2.0 — default 1.0 дає silhouette)
    if req.hdri_path.exists():
        bproc.world.set_world_background_hdr_img(str(req.hdri_path), strength=2.0)

    # 4. Explicit SUN light для daylight. HDRI сам по собі дає тільки ambient sky;
    # без directional sun vehicle виходить майже чорний при overcast HDRI.
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(5.0)
    # Sun direction: zenith angle 20-50° (~10am-3pm), random azimuth.
    # SUN type у Blender — directional, location ignored, тільки rotation.
    sun_zenith = math.radians(float(rng.uniform(20, 50)))
    sun_azim = float(rng.uniform(0, 2 * math.pi))
    sun.set_rotation_euler([sun_zenith, 0, sun_azim])

    # 5. Drone oblique/nadir геометрія:
    # distance_m = 3D line-of-sight камера→vehicle.
    # H = d·sin(θ), xy = d·cos(θ). cam_z = H > 0 → завжди look-down.
    elev_rad = math.radians(max(req.camera.view_angle_deg, 5.0))
    altitude = req.camera.distance_m * math.sin(elev_rad)
    distance_xy = req.camera.distance_m * math.cos(elev_rad)
    azimuth = float(rng.uniform(0, 2 * math.pi))
    cam_x = float(center[0]) + distance_xy * math.cos(azimuth)
    cam_y = float(center[1]) + distance_xy * math.sin(azimuth)
    cam_z = altitude
    cam_pose = np.array([cam_x, cam_y, cam_z], dtype=float)

    # При view_angle≈90° (nadir) forward vector майже паралельний світовому Z,
    # rotation_from_forward_vec(default up=[0,0,1]) дегенерує (cross product → 0).
    # Special case: будуємо rotation напряму через Euler. Blender camera default
    # дивиться вниз (-Z) при rotation_euler=(0,0,0), top-of-frame = +Y.
    # Z-axis rotation = azimuth для variety орієнтації кадру.
    #
    # Threshold 89° (НЕ 85°): rotation_from_forward_vec дегенерує лише в межах ~1°
    # від справжнього надіра. При 85° forward ще на 5° від вертикалі — look-at гілка
    # коректно цілиться у vehicle. А straight-down гілка ігнорує center: камера
    # зміщена на distance·cos(θ) (130 м при θ=85°,d=1500) і дивиться рівно вниз →
    # vehicle (off-axis 130 м) випадає за межі 6°-FOV footprint (~78 м півширина) →
    # порожній кадр. Тому straight-down лишаємо тільки для θ≥89°, де зміщення ≤22 м
    # (у межах footprint), і додатково ставимо камеру СТРОГО над vehicle (xy=center),
    # щоб надір гарантовано бачив техніку.
    NADIR_THRESHOLD_DEG = 89.0
    if req.camera.view_angle_deg >= NADIR_THRESHOLD_DEG:
        from mathutils import Euler
        cam_pose = np.array([float(center[0]), float(center[1]), altitude], dtype=float)
        rot_mat = Euler((0.0, 0.0, azimuth), 'XYZ').to_matrix()
        look_at_matrix = bproc.math.build_transformation_mat(cam_pose, rot_mat)
    else:
        look_at_matrix = bproc.math.build_transformation_mat(
            cam_pose,
            bproc.camera.rotation_from_forward_vec(center - cam_pose),
        )
    bproc.camera.add_camera_pose(look_at_matrix)

    # 6. Camera intrinsics
    bproc.camera.set_resolution(req.image_w, req.image_h)
    cam = bpy.context.scene.camera.data
    cam.lens = req.camera.focal_mm
    cam.sensor_width = req.camera.sensor_width_mm
    # КРИТИЧНО: Blender camera default clip_end = 1000 м. Camera→vehicle distance
    # (= distance_m) і ground-patch у FOV-конусі лежать на distance_m..(distance_m+horizon)
    # метрів. При distance_m=1500-2500 (iter5b small-vehicle 20-33px) ВСЕ за far-clip
    # 1000 м → Cycles обрізає всю геометрію → порожній кадр: vehicle_masks=0, n_boxes=0,
    # RGB=саме HDRI-небо, depth=65535 (no-hit background) скрізь. Розсуваємо far-plane
    # за найдальшу геометрію (ground plane 5000-scale = 10км край + горизонт HDRI).
    cam.clip_start = 0.1
    cam.clip_end = max(50000.0, req.camera.distance_m * 5.0)

    sun_info = {
        "sun_zenith_rad": float(sun_zenith),
        "sun_azim_rad": float(sun_azim),
        "sun_elevation_deg": float(math.degrees(math.pi / 2 - sun_zenith)),
        "sun_azimuth_deg": float(math.degrees(sun_azim) % 360.0),
    }
    return look_at_matrix, vehicle_meshes, sun_info
