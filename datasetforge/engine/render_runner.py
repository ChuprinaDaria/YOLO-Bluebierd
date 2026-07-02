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


def _vehicle_mask(data, class_id):
    """Бінарна маска (uint8 0/255) пікселів техніки з render output.

    map_by=["category_id", "instance"] дає окремий category_id_segmaps.
    """
    cat_segmaps = data.get("category_id_segmaps")
    if cat_segmaps:
        cat_arr = np.asarray(cat_segmaps[0])
        return (cat_arr == int(class_id)).astype(np.uint8) * 255
    # Fallback: filter instance_segmaps via attribute map.
    segmap = np.asarray(data["instance_segmaps"][0])
    inst_attrs = data["instance_attribute_maps"][0]
    veh_ids = [a.get("idx") for a in inst_attrs
               if int(a.get("category_id", -1)) == int(class_id)]
    veh_ids = [v for v in veh_ids if v is not None]
    return (np.isin(segmap, veh_ids).astype(np.uint8) * 255
            if veh_ids else np.zeros_like(segmap, dtype=np.uint8))


def main(argv=None):
    args = parse_args(argv)

    # bproc вже імпортований на верху файла (BlenderProc requirement).
    from datasetforge.engine.scene_builder import CameraSpec, SceneRequest, build_scene
    from datasetforge.engine.bbox_extractor import (
        mask_to_yolo_box, mask_to_yolo_obb, write_yolo_label, write_yolo_obb,
    )
    from datasetforge.engine.camera_sampler import (
        build_grid_from_config, estimate_target_px, filter_viable,
    )
    from datasetforge.engine.season_lighting import pick_season_assets
    from datasetforge.pipelines.inpaint.prompts import azimuth_to_cardinal

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cls_meta = cfg["class"]
    cam_cfg = cfg["camera"]
    scene_cfg = cfg["scene"]

    # Рендер у НАТИВНУ роздільність сенсора (1920×1080), не 640.
    sensor = cam_cfg.get("sensor", {}) or {}
    if "image_size" in cfg:
        img_w, img_h = cfg["image_size"]
    else:
        img_w, img_h = int(sensor.get("width", 1920)), int(sensor.get("height", 1080))

    target_size_m = float(cls_meta.get("target_size_m", 5.0))

    out_cfg = cfg.get("output", {}) or {}
    min_side_px = int(out_cfg.get("min_side_px", 6))
    bbox_format = str(out_cfg.get("bbox_format", "aabb")).lower()  # aabb | obb

    hn_cfg = cfg.get("hard_negatives", {}) or {}
    hn_ratio = float(hn_cfg.get("ratio", 0.0) or 0.0)
    hn_enabled = hn_ratio > 0 and "empty_landscape" in (hn_cfg.get("types") or [])
    hn_period = int(round(1.0 / hn_ratio)) if hn_enabled else 0

    # Оклюжн: дерева/кущі/сітки між камерою і ціллю (domain gap fix).
    occ_cfg = cfg.get("occlusion", {}) or {}
    occ_on = bool(occ_cfg.get("enabled", False))
    occ_ratio = float(occ_cfg.get("ratio", 0.5) or 0.0)
    occ_nrange = occ_cfg.get("n_range", [1, 4])
    occ_kinds = tuple(occ_cfg.get("kinds", ["tree", "bush"]))
    min_visible_frac = float(occ_cfg.get("min_visible_frac", 0.25))

    # Destroyed/wreck: вигоріла техніка. mode hn (без боксу) | class (свій id).
    dstr_cfg = cfg.get("destroyed", {}) or {}
    dstr_on = bool(dstr_cfg.get("enabled", False))
    dstr_ratio = float(dstr_cfg.get("ratio", 0.0) or 0.0)
    dstr_mode = str(dstr_cfg.get("mode", "hn")).lower()
    dstr_class_id = dstr_cfg.get("class_id", None)
    dstr_period = int(round(1.0 / dstr_ratio)) if (dstr_on and dstr_ratio > 0) else 0

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

    grid = build_grid_from_config(cam_cfg)
    viable_grid, rejected = filter_viable(grid, img_w, target_size_m, min_side_px)
    for s, est_min in rejected:
        print(f"[grid-skip] alt={s.altitude_m:.0f}m angle={s.view_angle_deg:.0f}° "
              f"hfov={s.hfov_deg:.0f}° (d={s.distance_m:.0f}m): est min-side "
              f"{est_min:.1f}px < {min_side_px}px — комбінація нежиттєздатна")
    print(f"[grid] {len(viable_grid)}/{len(grid)} viable combos "
          f"(image_w={img_w}, target={target_size_m}m, min_side={min_side_px}px)")
    print(f"[cfg] bbox_format={bbox_format} occlusion={'on' if occ_on else 'off'} "
          f"(ratio={occ_ratio}, min_visible={min_visible_frac}) "
          f"destroyed={'on:' + dstr_mode if dstr_on else 'off'}")
    if not viable_grid:
        print("[err] жодна комбінація камери не проходить pixel budget", file=sys.stderr)
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

    # Sesja 6 fix (перенесено після PR#3 rewrite): depth/normal enable — ODIN РАЗ
    # на весь run. BlenderProc падає RuntimeError якщо викликати двічі.
    # Segmentation має лишатися у loop (scene rebuild скидає pass index у BP 2.8).
    if args.depth_aov:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    if args.normal_aov:
        bproc.renderer.enable_normals_output()

    kept = 0
    discarded = 0
    hn_written = 0
    occ_written = 0
    wreck_written = 0
    attempt = 0
    max_attempts = max(args.n * 3, args.n + 20)  # оклюжн підвищує discard-rate

    while kept < args.n and attempt < max_attempts:
        i = attempt
        attempt += 1
        seed = args.seed + i
        rng_out = np.random.default_rng(seed)
        season = scene_cfg["seasons"][i % len(scene_cfg["seasons"])]
        landscape = scene_cfg["landscapes"][i % len(scene_cfg["landscapes"])]
        model_path = model_variants[i % len(model_variants)]
        assets = pick_season_assets(args.assets_root, season, seed=seed)
        cam_sample = viable_grid[i % len(viable_grid)]

        is_hn = hn_period > 0 and (kept % hn_period) == (hn_period - 1)
        is_wreck = (not is_hn) and dstr_period > 0 and (kept % dstr_period) == 0
        wreck_mode = dstr_mode if is_wreck else "off"
        # Клас мітки: wreck-class пише свій destroyed id; решта — клас техніки.
        label_class_id = int(cls_meta["id"])
        seg_class_id = int(cls_meta["id"])
        if is_wreck and dstr_mode == "class" and dstr_class_id is not None:
            label_class_id = int(dstr_class_id)
            seg_class_id = int(dstr_class_id)

        n_occ = 0
        if occ_on and not is_hn and rng_out.random() < occ_ratio:
            lo, hi = int(occ_nrange[0]), int(occ_nrange[1])
            n_occ = int(rng_out.integers(lo, hi + 1))

        est_long_px, est_min_px = estimate_target_px(cam_sample, img_w, target_size_m)

        camera = CameraSpec(
            distance_m=cam_sample.distance_m,
            view_angle_deg=cam_sample.view_angle_deg,
            hfov_deg=cam_sample.hfov_deg,
        )
        req = SceneRequest(
            class_name=cls_meta["name"],
            class_id=seg_class_id,
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
            n_occluders=n_occ,
            occluder_kinds=occ_kinds,
            wreck_mode=wreck_mode,
        )

        bproc.utility.reset_keyframes()
        _, _, sun_info, occluder_objs = build_scene(req)
        sun_cardinal = azimuth_to_cardinal(sun_info["sun_azimuth_deg"])

        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        # TWO-PASS visibility (тільки коли є оклюдери):
        #   pass A (amodal) — оклюдери приховані → ПОВНИЙ силует цілі M_full.
        #     Дешевий: 4 samples, без depth/normal — потрібна лише сегментація.
        #   pass B (occluded) — оклюдери видимі → ВИДИМИЙ силует M_vis + color+AOV.
        #   visibility = |M_vis| / |M_full|. bbox рахуємо з M_full (amodal, стабільний).
        mask_full = None
        if occluder_objs:
            for o in occluder_objs:
                o.blender_obj.hide_render = True
            bproc.renderer.set_max_amount_of_samples(4)
            data_amodal = bproc.renderer.render()
            mask_full = _vehicle_mask(data_amodal, seg_class_id)
            for o in occluder_objs:
                o.blender_obj.hide_render = False

        bproc.renderer.set_max_amount_of_samples(64)
        # depth/normals enable — ODIN РАЗ перед loop (див. рядки ~173). Тут НЕ дублювати.
        data = bproc.renderer.render()

        mask_vis = _vehicle_mask(data, seg_class_id)
        if mask_full is None:
            mask_full = mask_vis  # single-pass: amodal == visible

        vis_px = int((mask_vis > 0).sum())
        full_px = int((mask_full > 0).sum())
        visibility = (vis_px / full_px) if full_px > 0 else 0.0

        # amodal bbox з ПОВНОГО силуету (best practice — стабільний під оклюжном).
        if bbox_format == "obb":
            obb = (None if (is_hn or (is_wreck and dstr_mode == "hn"))
                   else mask_to_yolo_obb(mask_full, label_class_id, min_side_px))
            box = mask_to_yolo_box(mask_full, label_class_id, min_side_px) \
                if obb is not None else None  # aabb sanity-паралельно для фільтра
            has_box = obb is not None
        else:
            box = (None if (is_hn or (is_wreck and dstr_mode == "hn"))
                   else mask_to_yolo_box(mask_full, label_class_id, min_side_px))
            obb = None
            has_box = box is not None

        # --- Рішення keep / discard ---
        expect_empty = is_hn or (is_wreck and dstr_mode == "hn")
        if expect_empty:
            # Чесний негатив: техніки-цілі нема (hn) або брухт без боксу.
            boxes_out = []
        else:
            if not has_box:
                # Ціль поза кадром або amodal min-side < порога → discard.
                reason = (f"техніка видима ({full_px}px), amodal min-side < {min_side_px}px"
                          if full_px > 0 else "техніка поза кадром")
                print(f"[discard] frame#{i} alt={cam_sample.altitude_m:.0f}m "
                      f"angle={cam_sample.view_angle_deg:.0f}°: {reason}")
                discarded += 1
                continue
            if occluder_objs and visibility < min_visible_frac:
                # Перекрито сильніше за поріг видимості → шум, не сигнал.
                print(f"[discard] frame#{i}: visibility {visibility:.2f} < "
                      f"{min_visible_frac} (occluded {100*(1-visibility):.0f}%)")
                discarded += 1
                continue
            boxes_out = [box] if bbox_format != "obb" else [obb]

        stem = f"{cls_meta['name']}_{kept:05d}"

        # Color image
        colors = data.get("colors")
        if colors:
            rgb = np.asarray(colors[0])
            if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                bgr = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            print(f"[warn] no color output frame {i}", file=sys.stderr)

        # Label (aabb або obb)
        if bbox_format == "obb":
            write_yolo_obb([b for b in boxes_out if b is not None],
                           lbl_dir / f"{stem}.txt")
        else:
            write_yolo_label([b for b in boxes_out if b is not None],
                             lbl_dir / f"{stem}.txt")

        # depth AOV
        depth_scale = 65535.0 / max(cam_sample.distance_m * 1.5, 1.0)
        if args.depth_aov and "depth" in data and data["depth"]:
            depth_arr = np.asarray(data["depth"][0], dtype=np.float32)
            depth_u16 = np.clip(depth_arr * depth_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_u16)

        # normals AOV
        if args.normal_aov and "normals" in data and data["normals"]:
            normals_arr = np.asarray(data["normals"][0], dtype=np.float32)
            normals_u16 = np.clip(((normals_arr + 1.0) * 0.5) * 65535.0,
                                  0, 65535).astype(np.uint16)
            cv2.imwrite(str(normal_dir / f"{stem}.png"), normals_u16)

        # vehicle mask = ВИДИМІ пікселі (для diffusion freeze реальної геометрії).
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask_vis)

        focal_px = (img_w / 2.0) / math.tan(math.radians(cam_sample.hfov_deg) / 2.0)
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
            "class_id": label_class_id,
            "target_size_m": target_size_m,
            "est_target_px": [round(est_long_px, 1), round(est_min_px, 1)],
            "min_side_px": min_side_px,
            "bbox_format": bbox_format,
            "n_boxes": len([b for b in boxes_out if b is not None]),
            "has_targets": len([b for b in boxes_out if b is not None]) > 0,
            "is_hard_negative": is_hn,
            "is_destroyed": is_wreck,
            "wreck_mode": wreck_mode,
            "n_occluders": n_occ,
            "occluder_kinds": list(occ_kinds) if n_occ else [],
            "visibility_fraction": round(visibility, 4),
            "is_occluded": bool(occluder_objs) and visibility < 0.98,
            "vehicle_visible_px": vis_px,
            "vehicle_full_px": full_px,
            "model_variant": None if is_hn else model_path.name,
            "hdri": assets.hdri.name,
            "ground_texture": assets.ground_texture.name,
            "sun_azimuth_deg": sun_info["sun_azimuth_deg"],
            "sun_elevation_deg": sun_info["sun_elevation_deg"],
            "sun_cardinal": sun_cardinal,
            "camera_intrinsics": {
                "fx": float(focal_px), "fy": float(focal_px),
                "cx": float(img_w / 2.0), "cy": float(img_h / 2.0),
            },
            "vehicle_category_ids": [seg_class_id],
            "depth_scale_mm_per_unit": float(depth_scale),
            "diffusion": {"enabled": False},
        }
        (meta_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        kept += 1
        if is_hn:
            hn_written += 1
        if is_wreck:
            wreck_written += 1
        if n_occ:
            occ_written += 1
        if is_hn:
            kind = "HARD-NEG"
        elif is_wreck:
            kind = f"WRECK:{dstr_mode}"
        else:
            kind = f"boxes={meta['n_boxes']}"
        occ_note = (f" occ={n_occ}/vis={visibility:.2f}" if n_occ else "")
        print(f"[{kept}/{args.n}] alt={cam_sample.altitude_m:.0f}m d={cam_sample.distance_m:.0f}m "
              f"angle={cam_sample.view_angle_deg:.0f}° hfov={cam_sample.hfov_deg:.0f}° "
              f"est={est_long_px:.0f}×{est_min_px:.0f}px {kind}{occ_note} "
              f"season={season} sun={sun_cardinal}")

    if kept < args.n:
        print(f"[warn] kept={kept} < n={args.n} після {attempt} спроб "
              f"(discarded={discarded}) — перевір grid/occlusion/min_side", file=sys.stderr)
    print(f"[done] kept={kept} (hard_neg={hn_written} wreck={wreck_written} "
          f"occluded={occ_written}) discarded={discarded} attempts={attempt} → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
