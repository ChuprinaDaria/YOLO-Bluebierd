import blenderproc as bproc  # MUST be first line per BlenderProc CLI check.

# Per-config render orchestrator. Runs only via `blenderproc run <this_file>`.

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# Add repo root до sys.path щоб `datasetforge.engine.*` імпортувалось.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import yaml


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20, help="кількість кадрів")
    ap.add_argument("--out", type=Path, required=True, help="вихідна папка")
    ap.add_argument("--assets-root", type=Path, default=Path("datasetforge/assets"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth-aov", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable depth AOV → out/depth/{stem}.png (16-bit, depth_mm)")
    ap.add_argument("--normal-aov", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable normals AOV → out/normals/{stem}.png (16-bit 3ch, encoded)")
    ap.add_argument("--render-only", action=argparse.BooleanOptionalAction, default=True,
                    help="Render-only. Stage 3+4 (diffusion+composite) запускаються окремо.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # bproc вже імпортований на верху файла (BlenderProc requirement).
    from datasetforge.engine.scene_builder import CameraSpec, SceneRequest, build_scene
    from datasetforge.engine.bbox_extractor import coco_to_yolo, write_yolo_label
    from datasetforge.engine.camera_sampler import build_grid, sample_round_robin
    from datasetforge.engine.season_lighting import pick_season_assets
    from datasetforge.pipelines.inpaint.prompts import azimuth_to_cardinal

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cls_meta = cfg["class"]
    img_w, img_h = cfg["image_size"]
    cam_cfg = cfg["camera"]
    scene_cfg = cfg["scene"]

    bproc.init()

    # GPU enable (Cycles CUDA / OptiX).
    try:
        import bpy
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                bpy.context.scene.cycles.device = "GPU"
                print(f"[gpu] Cycles backend={backend}, devices={[d.name for d in prefs.devices if d.use]}")
                break
            except Exception as e:
                print(f"[gpu] {backend} failed: {e}")
        else:
            print("[gpu] не вдалось увімкнути GPU, лишаємось на CPU")
    except Exception as e:
        print(f"[gpu] GPU setup error (продовжуємо CPU): {e}")

    grid = build_grid(cam_cfg["distance_m"], cam_cfg["view_angle_deg"])
    samples = sample_round_robin(grid, args.n)

    out_dir = args.out
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    meta_dir = out_dir / "metadata"
    depth_dir = out_dir / "depth"
    normal_dir = out_dir / "normals"
    mask_dir = out_dir / "vehicle_masks"
    output_dirs = [img_dir, lbl_dir, meta_dir, mask_dir]
    if args.depth_aov:
        output_dirs.append(depth_dir)
    if args.normal_aov:
        output_dirs.append(normal_dir)
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    models_dir = args.assets_root / "models" / cls_meta["name"]
    model_variants = []
    for ext in ("*.blend", "*.glb", "*.gltf", "*.fbx", "*.obj"):
        model_variants.extend(models_dir.glob(ext))
    model_variants = sorted(model_variants)
    if not model_variants:
        print(f"[err] no .blend/.glb/.gltf/.fbx/.obj in {models_dir}", file=sys.stderr)
        return 2
    print(f"[models] {len(model_variants)} variants: {[m.name for m in model_variants]}")

    # Camera intrinsics — однакові для всіх кадрів (hfov + image_size фіксовані).
    focal_px = (img_w / 2.0) / math.tan(math.radians(cam_cfg["hfov_deg"]) / 2.0)
    intrinsics = {
        "fx": float(focal_px),
        "fy": float(focal_px),
        "cx": float(img_w / 2.0),
        "cy": float(img_h / 2.0),
    }

    for i, cam_sample in enumerate(samples):
        seed = args.seed + i
        season = scene_cfg["seasons"][i % len(scene_cfg["seasons"])]
        landscape = scene_cfg["landscapes"][i % len(scene_cfg["landscapes"])]
        model_path = model_variants[i % len(model_variants)]
        assets = pick_season_assets(args.assets_root, season, seed=seed)

        camera = CameraSpec(
            distance_m=cam_sample.distance_m,
            view_angle_deg=cam_sample.view_angle_deg,
            hfov_deg=cam_cfg["hfov_deg"],
        )
        req = SceneRequest(
            class_name=cls_meta["name"],
            class_id=cls_meta["id"],
            model_path=model_path,
            hdri_path=assets.hdri,
            ground_texture_path=assets.ground_texture,
            camera=camera,
            season=season,
            landscape=landscape,
            weather=scene_cfg["weather"][i % len(scene_cfg["weather"])],
            image_w=img_w,
            image_h=img_h,
            seed=seed,
            road_under_vehicle=bool(scene_cfg.get("road_under_vehicle", False)),
        )

        bproc.utility.reset_keyframes()
        _, _, sun_info = build_scene(req)
        sun_cardinal = azimuth_to_cardinal(sun_info["sun_azimuth_deg"])

        # Depth scale підбираємо ПІД дальність кадру. Фіксований ×1000 (мм) саттурив
        # uint16 (65535 = 65.5м) на distance 1500-2500м → depth=65535 скрізь →
        # cond.py percentile-norm падав (lo==hi) → depth-conditioning мертвий.
        # 65535 / (distance·1.5) лишає техніку (~distance) і ближню землю у градієнті;
        # далека земля/небо клипиться у 65535 (cond.py все одно ріже 1-99 перцентиль).
        depth_scale = 65535.0 / max(cam_sample.distance_m * 1.5, 1.0)

        bproc.renderer.set_max_amount_of_samples(64)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])
        if args.depth_aov:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        if args.normal_aov:
            bproc.renderer.enable_normals_output()
        data = bproc.renderer.render()

        # Тимчасова COCO папка, потім конвертуємо у YOLO
        tmp_coco = out_dir / f"_coco_{i:05d}"
        tmp_coco.mkdir(exist_ok=True)
        bproc.writer.write_coco_annotations(
            str(tmp_coco),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
            jpg_quality=85,
            label_mapping=bproc.utility.LabelIdMapping.from_dict({cls_meta["name"]: cls_meta["id"]}),
        )

        for img_filename, boxes in coco_to_yolo(
            tmp_coco / "coco_annotations.json",
            image_w=img_w, image_h=img_h, min_side_px=10,
        ):
            stem = f"{cls_meta['name']}_{i:05d}"
            dst_img = img_dir / f"{stem}.jpg"
            src_img = tmp_coco / img_filename
            if not src_img.exists():
                candidates = list(tmp_coco.rglob("*.jpg")) + list(tmp_coco.rglob("*.jpeg")) + list(tmp_coco.rglob("*.png"))
                if candidates:
                    src_img = candidates[0]
            if src_img.exists():
                shutil.move(str(src_img), dst_img)
            else:
                print(f"[warn] no image source for frame {i}; checked {tmp_coco}", file=sys.stderr)
            write_yolo_label(boxes, lbl_dir / f"{stem}.txt")

            # AOV: depth → 16-bit PNG, per-frame depth_scale (sidecar). cond.py
            # percentile-нормалізує → scale-invariant; головне не саттурити 65535.
            if args.depth_aov and "depth" in data and data["depth"]:
                depth_arr = np.asarray(data["depth"][0], dtype=np.float32)
                depth_u16 = np.clip(depth_arr * depth_scale, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_u16)

            # AOV: normals → 16-bit 3ch PNG, encoded (n+1)/2 * 65535. cv2 round-trip
            # bit-preserves array; composite reads back via cv2.imread same ordering.
            if args.normal_aov and "normals" in data and data["normals"]:
                normals_arr = np.asarray(data["normals"][0], dtype=np.float32)
                normals_u16 = np.clip(((normals_arr + 1.0) * 0.5) * 65535.0,
                                      0, 65535).astype(np.uint16)
                cv2.imwrite(str(normal_dir / f"{stem}.png"), normals_u16)

            # Vehicle alpha mask: pick pixels where category_id == our class.
            # map_by=["category_id", "instance"] дає category_id_segmaps окремий.
            cat_segmaps = data.get("category_id_segmaps")
            if cat_segmaps:
                cat_arr = np.asarray(cat_segmaps[0])
                mask = (cat_arr == int(cls_meta["id"])).astype(np.uint8) * 255
            else:
                # Fallback: filter instance_segmaps via attribute map.
                segmap = np.asarray(data["instance_segmaps"][0])
                inst_attrs = data["instance_attribute_maps"][0]
                veh_ids = [a.get("idx") for a in inst_attrs
                           if int(a.get("category_id", -1)) == int(cls_meta["id"])]
                veh_ids = [v for v in veh_ids if v is not None]
                mask = (np.isin(segmap, veh_ids).astype(np.uint8) * 255
                        if veh_ids else np.zeros_like(segmap, dtype=np.uint8))
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

            # Metadata sidecar
            _elev_rad = math.radians(max(cam_sample.view_angle_deg, 5.0))
            meta = {
                "image_id": stem,
                "source": "synthetic_blender",
                "dataset_version": cfg.get("version", "df-v1.0.0-dev"),
                "seed": seed,
                "distance_m": cam_sample.distance_m,
                "altitude_m": cam_sample.distance_m * math.sin(_elev_rad),
                "horizontal_distance_m": cam_sample.distance_m * math.cos(_elev_rad),
                "view_angle_deg": cam_sample.view_angle_deg,
                "hfov_deg": cam_cfg["hfov_deg"],
                "sensor_res": [img_w, img_h],
                "modality": "EO",
                "season": season,
                "landscape": landscape,
                "weather": req.weather,
                "time_of_day": "day",
                "class_name": cls_meta["name"],
                "class_id": cls_meta["id"],
                "n_boxes": len(boxes),
                "has_targets": len(boxes) > 0,
                "model_variant": model_path.name,
                "hdri": assets.hdri.name,
                "ground_texture": assets.ground_texture.name,
                "sun_azimuth_deg": sun_info["sun_azimuth_deg"],
                "sun_elevation_deg": sun_info["sun_elevation_deg"],
                "sun_cardinal": sun_cardinal,
                "camera_intrinsics": intrinsics,
                "vehicle_category_ids": [cls_meta["id"]],
                "depth_scale_mm_per_unit": float(depth_scale),
                "diffusion": {"enabled": False},
            }
            (meta_dir / f"{stem}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        shutil.rmtree(tmp_coco, ignore_errors=True)
        _alt = cam_sample.distance_m * math.sin(math.radians(max(cam_sample.view_angle_deg, 5.0)))
        print(f"[{i+1}/{args.n}] d={cam_sample.distance_m:.0f}m angle={cam_sample.view_angle_deg:.0f}° "
              f"alt={_alt:.0f}m season={season} model={model_path.name} sun={sun_cardinal}")

    print(f"[done] {args.n} frames → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
