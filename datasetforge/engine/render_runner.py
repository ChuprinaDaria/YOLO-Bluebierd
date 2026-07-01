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
    ap.add_argument("--n", type=int, default=20,
                    help="кількість кадрів У ВИХОДІ (discard-и добираються зверху)")
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
    from datasetforge.engine.camera_sampler import (
        build_grid_from_config, estimate_target_px, filter_viable,
    )
    from datasetforge.engine.season_lighting import pick_season_assets
    from datasetforge.pipelines.inpaint.prompts import azimuth_to_cardinal

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cls_meta = cfg["class"]
    cam_cfg = cfg["camera"]
    scene_cfg = cfg["scene"]

    # Рендер у НАТИВНУ роздільність сенсора (1920×1080), не 640: інакше GSD у
    # 3× гірший за реальну камеру і вся техніка — «крапки». image_size у YAML
    # має явний пріоритет; fallback — camera.sensor.
    sensor = cam_cfg.get("sensor", {}) or {}
    if "image_size" in cfg:
        img_w, img_h = cfg["image_size"]
    else:
        img_w, img_h = int(sensor.get("width", 1920)), int(sensor.get("height", 1080))

    # Реальна довжина техніки (нормалізація моделі), per-class з YAML.
    target_size_m = float(cls_meta.get("target_size_m", 5.0))

    out_cfg = cfg.get("output", {}) or {}
    min_side_px = int(out_cfg.get("min_side_px", 6))

    hn_cfg = cfg.get("hard_negatives", {}) or {}
    hn_ratio = float(hn_cfg.get("ratio", 0.0) or 0.0)
    hn_enabled = hn_ratio > 0 and "empty_landscape" in (hn_cfg.get("types") or [])
    hn_period = int(round(1.0 / hn_ratio)) if hn_enabled else 0

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

    # Camera grid: altitude × view_angle × hfov (92/112). Preflight pixel budget
    # викидає комбінації, де мін-сторона bbox гарантовано < min_side_px — той
    # самий поріг, що post-render фільтр, тому «рендер → одразу discard» не буває.
    grid = build_grid_from_config(cam_cfg)
    viable_grid, rejected = filter_viable(grid, img_w, target_size_m, min_side_px)
    for s, est_min in rejected:
        print(f"[grid-skip] alt={s.altitude_m:.0f}m angle={s.view_angle_deg:.0f}° "
              f"hfov={s.hfov_deg:.0f}° (d={s.distance_m:.0f}m): est min-side "
              f"{est_min:.1f}px < {min_side_px}px — комбінація нежиттєздатна")
    print(f"[grid] {len(viable_grid)}/{len(grid)} viable combos "
          f"(image_w={img_w}, target={target_size_m}m, min_side={min_side_px}px)")
    if not viable_grid:
        print("[err] жодна комбінація камери не проходить pixel budget — "
              "збільш image_size або зменш висоти/hfov", file=sys.stderr)
        return 2

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

    kept = 0          # записані кадри (vehicle + hard negative)
    discarded = 0     # кадри, викинуті post-render перевіркою
    hn_written = 0
    attempt = 0
    # Добираємо discard-и зверху, але з жорсткою стелею щоб не крутитись вічно.
    max_attempts = max(args.n * 2, args.n + 10)

    while kept < args.n and attempt < max_attempts:
        i = attempt
        attempt += 1
        seed = args.seed + i
        season = scene_cfg["seasons"][i % len(scene_cfg["seasons"])]
        landscape = scene_cfg["landscapes"][i % len(scene_cfg["landscapes"])]
        model_path = model_variants[i % len(model_variants)]
        assets = pick_season_assets(args.assets_root, season, seed=seed)
        cam_sample = viable_grid[i % len(viable_grid)]

        # Hard negative slot: кожен hn_period-ний ВИХІДНИЙ кадр — порожній
        # ландшафт без техніки (справжній negative, а не жертва фільтра).
        is_hn = hn_period > 0 and (kept % hn_period) == (hn_period - 1)

        est_long_px, est_min_px = estimate_target_px(cam_sample, img_w, target_size_m)

        camera = CameraSpec(
            distance_m=cam_sample.distance_m,
            view_angle_deg=cam_sample.view_angle_deg,
            hfov_deg=cam_sample.hfov_deg,
        )
        req = SceneRequest(
            class_name=cls_meta["name"],
            class_id=cls_meta["id"],
            model_path=None if is_hn else model_path,
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
            target_size_m=target_size_m,
            hard_negative=is_hn,
        )

        bproc.utility.reset_keyframes()
        _, _, sun_info = build_scene(req)
        sun_cardinal = azimuth_to_cardinal(sun_info["sun_azimuth_deg"])

        # Depth scale підбираємо ПІД дальність кадру. Фіксований ×1000 (мм) саттурив
        # uint16 (65535 = 65.5м) на далеких дистанціях → depth=65535 скрізь →
        # cond.py percentile-norm падав (lo==hi) → depth-conditioning мертвий.
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
        # quality 95: єдина «камерна» JPEG-компресія робиться ОДИН раз у Stage 5
        # polish; q85 тут + q60-85 там = подвійний JPEG, який стирає 10-20px цілі.
        bproc.writer.write_coco_annotations(
            str(tmp_coco),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
            jpg_quality=95,
            label_mapping=bproc.utility.LabelIdMapping.from_dict({cls_meta["name"]: cls_meta["id"]}),
        )

        entries = coco_to_yolo(
            tmp_coco / "coco_annotations.json",
            image_w=img_w, image_h=img_h, min_side_px=min_side_px,
        )
        if not entries:
            print(f"[discard] frame#{i}: COCO без images (render fail?)", file=sys.stderr)
            shutil.rmtree(tmp_coco, ignore_errors=True)
            discarded += 1
            continue
        img_filename, boxes, n_dropped = entries[0]

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

        # КРИТИЧНО: кадр з видимою технікою, але без жодного bbox (все відфільтровано
        # min-side, або техніка за кадром) — НЕ пишемо. Порожній label при видимій
        # техніці = false negative, який вчить модель ігнорувати цілі.
        if not is_hn and not boxes:
            mask_px = int((mask > 0).sum())
            reason = (f"техніка видима ({mask_px}px у масці), але всі {n_dropped} bbox "
                      f"< {min_side_px}px" if mask_px > 0 else "техніка поза кадром")
            print(f"[discard] frame#{i} alt={cam_sample.altitude_m:.0f}m "
                  f"angle={cam_sample.view_angle_deg:.0f}° hfov={cam_sample.hfov_deg:.0f}°: {reason}")
            shutil.rmtree(tmp_coco, ignore_errors=True)
            discarded += 1
            continue

        stem = f"{cls_meta['name']}_{kept:05d}"
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

        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

        # Camera intrinsics — per-frame, бо hfov тепер варіюється (92/112).
        focal_px = (img_w / 2.0) / math.tan(math.radians(cam_sample.hfov_deg) / 2.0)

        # Metadata sidecar
        meta = {
            "image_id": stem,
            "source": "synthetic_blender",
            "dataset_version": cfg.get("version", "df-v1.0.0-dev"),
            "seed": seed,
            "altitude_m": cam_sample.altitude_m,
            "distance_m": cam_sample.distance_m,
            "horizontal_distance_m": cam_sample.distance_m
                * math.cos(math.radians(max(cam_sample.view_angle_deg, 5.0))),
            "view_angle_deg": cam_sample.view_angle_deg,
            "hfov_deg": cam_sample.hfov_deg,
            "sensor_res": [img_w, img_h],
            "modality": sensor.get("modality", "EO"),
            "season": season,
            "landscape": landscape,
            "weather": req.weather,
            "time_of_day": "day",
            "class_name": cls_meta["name"],
            "class_id": cls_meta["id"],
            "target_size_m": target_size_m,
            "est_target_px": [round(est_long_px, 1), round(est_min_px, 1)],
            "min_side_px": min_side_px,
            "n_boxes": len(boxes),
            "n_dropped_subthreshold": n_dropped,
            "has_targets": len(boxes) > 0,
            "is_hard_negative": is_hn,
            "model_variant": None if is_hn else model_path.name,
            "hdri": assets.hdri.name,
            "ground_texture": assets.ground_texture.name,
            "sun_azimuth_deg": sun_info["sun_azimuth_deg"],
            "sun_elevation_deg": sun_info["sun_elevation_deg"],
            "sun_cardinal": sun_cardinal,
            "camera_intrinsics": {
                "fx": float(focal_px),
                "fy": float(focal_px),
                "cx": float(img_w / 2.0),
                "cy": float(img_h / 2.0),
            },
            "vehicle_category_ids": [cls_meta["id"]],
            "depth_scale_mm_per_unit": float(depth_scale),
            "diffusion": {"enabled": False},
        }
        (meta_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        shutil.rmtree(tmp_coco, ignore_errors=True)
        kept += 1
        if is_hn:
            hn_written += 1
        kind = "HARD-NEG" if is_hn else f"boxes={len(boxes)}"
        print(f"[{kept}/{args.n}] alt={cam_sample.altitude_m:.0f}m d={cam_sample.distance_m:.0f}m "
              f"angle={cam_sample.view_angle_deg:.0f}° hfov={cam_sample.hfov_deg:.0f}° "
              f"est={est_long_px:.0f}×{est_min_px:.0f}px {kind} "
              f"season={season} model={'—' if is_hn else model_path.name} sun={sun_cardinal}")

    if kept < args.n:
        print(f"[warn] kept={kept} < requested n={args.n} після {attempt} спроб "
              f"(discarded={discarded}) — перевір grid/min_side_px", file=sys.stderr)
    print(f"[done] kept={kept} (hard_neg={hn_written}) discarded={discarded} "
          f"attempts={attempt} → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
