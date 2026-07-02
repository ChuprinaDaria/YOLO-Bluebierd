"""Оклюдери: дерева / кущі / масксітки між камерою і технікою.

Чому: у реальному бойовому відео техніка стоїть у посадках, під кронами, з
маскувальними сітками — частину силуету перекрито. Стара сцена (техніка на
голому полі + заморожена маска у Flux) НІКОЛИ не давала оклюжна → domain gap:
модель провалюється на реально перекритих цілях.

Геометрія розміщення — pure-Python (тестується без bpy). Побудова 3D-примітивів
(`build_occluders`) робить lazy-import bproc/bpy — тільки під `blenderproc run`.

Best practice (arXiv 2101.08845 occlusion review; VOD-UAV occlusion levels):
  - оклюдери СТАВИМО НА ЛІНІЮ ЗОРУ камера→ціль, близько до цілі, щоб реально
    перекрити частину проєкції (а не просто «поряд у кадрі»);
  - видимість рахуємо з РЕНДЕРА (сегментація), не з геометрії — точно;
  - bbox лишаємо amodal (повний силует), фільтр — за visibility fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Оклюдери — category_id 0 (background): вони НЕ ціль, не потрапляють у vehicle-маску,
# але фізично перекривають техніку у рендері.
OCCLUDER_CATEGORY_ID = 0


@dataclass
class OccluderSpec:
    kind: str          # "tree" | "bush" | "net"
    x: float
    y: float
    z: float           # base z (низ об'єкта на землі, z=0)
    height_m: float    # повна висота
    radius_m: float    # півширина крони/куща/сітки
    z_rot_rad: float   # yaw для variety


def plan_occluders(
    vehicle_xy: tuple[float, float],
    camera_xy: tuple[float, float],
    rng,
    *,
    n: int,
    kinds: list[str],
    gap_m: tuple[float, float] = (1.5, 8.0),
    lateral_m: tuple[float, float] = (-2.0, 2.0),
    tree_h: tuple[float, float] = (8.0, 14.0),
    bush_h: tuple[float, float] = (1.5, 3.5),
    net_h: tuple[float, float] = (2.5, 4.0),
) -> list[OccluderSpec]:
    """Розкидати `n` оклюдерів між технікою і камерою.

    rng — np.random.Generator (або сумісний .uniform / .choice / .integers).
    Кожен оклюдер: на промені vehicle→camera на відстані gap від техніки +
    боковий зсув lateral (мале зміщення = більше перекриття).
    """
    vx, vy = float(vehicle_xy[0]), float(vehicle_xy[1])
    cx, cy = float(camera_xy[0]), float(camera_xy[1])
    dx, dy = cx - vx, cy - vy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        # Надір: камера строго над ціллю — «між» нема; кладемо навколо по колу.
        ux, uy = 1.0, 0.0
        px, py = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist          # напрям до камери (уздовж зору)
        px, py = -uy, ux                        # перпендикуляр (боковий зсув)

    specs: list[OccluderSpec] = []
    for _ in range(n):
        kind = str(rng.choice(kinds))
        gap = float(rng.uniform(*gap_m))
        lat = float(rng.uniform(*lateral_m))
        ox = vx + ux * gap + px * lat
        oy = vy + uy * gap + py * lat
        if kind == "tree":
            h = float(rng.uniform(*tree_h))
            r = float(rng.uniform(3.0, 5.0))     # реалістична крона 3-5м для aerial view
        elif kind == "net":
            h = float(rng.uniform(*net_h))
            r = float(rng.uniform(3.0, 5.5))     # маскувальна сітка ~5-11м у діаметрі
        else:  # bush
            h = float(rng.uniform(*bush_h))
            r = float(rng.uniform(1.5, 3.0))     # bushy foliage ~3-6м
        specs.append(OccluderSpec(
            kind=kind, x=ox, y=oy, z=0.0,
            height_m=h, radius_m=r,
            z_rot_rad=float(rng.uniform(0, 2 * math.pi)),
        ))
    return specs


# Module-level cache: templates імпортуються ОДИН РАЗ, всі occluder-instance —
# linked copy (obj.copy() + shared mesh data). BlenderProc reset_keyframes зберігає
# hidden templates (z=-10000), тому cache живий upon весь run.
_TREE_TEMPLATES: list | None = None
_TREE_PACK_TRIED: bool = False


def _load_tree_templates(assets_root):
    """Імпорт `props/vegetation/low_poly_forest_tree_pack.glb` як hidden templates.

    Sesja 3 pattern: pack містить компоненти (Tree_Trunk_*, Tree_Branches_*,
    Background_Tree_Atlas_*). Беремо лише Background_Tree_Atlas — це повне дерево
    як crossed-planes billboard, цілісне. Окремі Tree_Trunk/Branches дадуть обрубки.

    Returns list of hidden template Blender objects (parked at z=-10000).
    Empty list if pack не знайдено (fallback до primitive у _build_tree).
    """
    global _TREE_TEMPLATES, _TREE_PACK_TRIED
    if _TREE_TEMPLATES is not None:
        return _TREE_TEMPLATES
    if _TREE_PACK_TRIED:
        return []  # уже пробували, немає
    _TREE_PACK_TRIED = True

    from pathlib import Path
    import bpy

    pack_path = Path(assets_root) / "props" / "vegetation" / "low_poly_forest_tree_pack.glb"
    if not pack_path.exists():
        print(f"[occluder] tree pack не знайдено: {pack_path} — fallback до primitive cone")
        _TREE_TEMPLATES = []
        return []

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(pack_path))
    new_names = list(set(bpy.data.objects.keys()) - before)

    keep = []
    for n in new_names:
        obj = bpy.data.objects.get(n)
        if obj is None:
            continue
        if obj.type != "MESH" or not obj.name.startswith("Background_Tree_Atlas"):
            # Обрубки Trunk/Branches окремо / EMPTY roots / Rocks — прибираємо.
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        # Normalize до 8м (типовий дуб/тополя UA).
        min_c = [min(v[i] for v in [obj.matrix_world @ vv.co for vv in obj.data.vertices]) for i in range(3)]
        max_c = [max(v[i] for v in [obj.matrix_world @ vv.co for vv in obj.data.vertices]) for i in range(3)]
        h = max_c[2] - min_c[2]
        if h < 0.1:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        s = 8.0 / h
        obj.scale = (s, s, s)
        obj.location = (0.0, 0.0, -10000.0)  # park hidden під землею
        obj["_df_tree_template"] = 1
        obj.hide_render = True  # templates НЕ рендеряться
        keep.append(obj)

    print(f"[occluder] tree templates loaded: {len(keep)} Background_Tree_Atlas variants")
    _TREE_TEMPLATES = keep
    return keep


def build_occluders(specs: list[OccluderSpec], season: str, assets_root=None):
    """Побудувати 3D-примітиви оклюдерів у сцені. Lazy bproc/bpy.

    tree kind — asset-based (real .glb billboard tree з low_poly_forest_tree_pack).
    bush/net — procedural (SPHERE / PLANE).

    Повертає список bproc MeshObject (щоб render_runner міг hide/unhide для
    two-pass visibility). Усі — category_id 0 (background).
    """
    import blenderproc as bproc

    # Сезонний колір крони/куща (взимку голіше/сіріше).
    if season == "winter":
        foliage = [0.28, 0.26, 0.22, 1.0]
    elif season == "autumn_mud":
        foliage = [0.35, 0.27, 0.12, 1.0]
    else:
        foliage = [0.12, 0.22, 0.09, 1.0]
    trunk = [0.20, 0.14, 0.09, 1.0]
    net_col = [0.24, 0.26, 0.20, 1.0]

    tree_templates = _load_tree_templates(assets_root) if assets_root else []

    objs = []
    for i, s in enumerate(specs):
        try:
            if s.kind == "tree":
                if tree_templates:
                    objs += _build_tree_from_template(bproc, s, tree_templates, i)
                else:
                    objs += _build_tree(bproc, s, foliage, trunk, i)
            elif s.kind == "net":
                objs.append(_build_net(bproc, s, net_col, i))
            else:
                objs.append(_build_bush(bproc, s, foliage, i))
        except Exception as exc:  # оклюдер — nice-to-have, не валимо рендер
            print(f"[occluder] skip {s.kind}#{i} (non-fatal): "
                  f"{exc.__class__.__name__}: {exc}")
    print(f"[occluder] built {len(objs)} meshes from {len(specs)} specs "
          f"(kinds={[s.kind for s in specs]})")
    return objs


def _build_tree_from_template(bproc, s: OccluderSpec, templates, idx):
    """Linked-copy tree template до позиції spec. Memory-efficient.

    obj.copy() з обмінянним mesh data (obj.data = src.data) — instance,
    геометрія shared. Тільки transform унікальний.
    """
    import bpy
    import random

    src = random.choice(templates)
    obj = src.copy()
    obj.data = src.data  # LINKED mesh — не новий datablock
    # Scale = ratio до template 8м → бажана height_m
    scale = float(s.height_m) / 8.0
    obj.scale = (scale, scale, scale)
    obj.location = (float(s.x), float(s.y), 0.0)
    obj.rotation_euler = (0.0, 0.0, float(s.z_rot_rad))
    obj.hide_render = False  # instance рендериться (template — ні)
    bpy.context.scene.collection.objects.link(obj)
    # Wrap у bproc MeshObject щоб render_runner міг hide_render toggle
    mo = bproc.types.MeshObject(obj)
    mo.set_cp("category_id", OCCLUDER_CATEGORY_ID)
    return [mo]


def _mat(bproc, name, color, rough=0.9):
    m = bproc.material.create(name)
    m.set_principled_shader_value("Base Color", color)
    m.set_principled_shader_value("Roughness", rough)
    # Blender 4.2 перейменував "Specular" → "Specular IOR Level". Пробуємо обидва;
    # для occluder-а це косметика, тому мовчки skip якщо ані немає.
    for key in ("Specular IOR Level", "Specular"):
        try:
            m.set_principled_shader_value(key, 0.05)
            break
        except (KeyError, RuntimeError):
            continue
    return m


def _build_tree(bproc, s: OccluderSpec, foliage, trunk, idx):
    trunk_h = s.height_m * 0.4
    crown_h = s.height_m - trunk_h
    stem = bproc.object.create_primitive(
        "CYLINDER", radius=max(0.12, s.radius_m * 0.15), depth=trunk_h)
    stem.set_location([s.x, s.y, trunk_h / 2.0])
    stem.set_cp("category_id", OCCLUDER_CATEGORY_ID)
    stem.replace_materials(_mat(bproc, f"occ_trunk_{idx}", trunk, rough=1.0))
    crown = bproc.object.create_primitive("CONE", radius=s.radius_m, depth=crown_h)
    crown.set_location([s.x, s.y, trunk_h + crown_h / 2.0])
    crown.set_rotation_euler([0.0, 0.0, s.z_rot_rad])
    crown.set_cp("category_id", OCCLUDER_CATEGORY_ID)
    crown.replace_materials(_mat(bproc, f"occ_crown_{idx}", foliage))
    return [stem, crown]


def _build_bush(bproc, s: OccluderSpec, foliage, idx):
    bush = bproc.object.create_primitive("SPHERE", radius=s.radius_m)
    # Squash у напівсферу-кущ: масштаб по Z менший, низ на землі.
    bush.set_location([s.x, s.y, s.height_m * 0.5])
    bush.set_scale([1.0, 1.0, max(0.4, s.height_m / (2.0 * s.radius_m))])
    bush.set_rotation_euler([0.0, 0.0, s.z_rot_rad])
    bush.set_cp("category_id", OCCLUDER_CATEGORY_ID)
    bush.replace_materials(_mat(bproc, f"occ_bush_{idx}", foliage))
    return bush


def _build_net(bproc, s: OccluderSpec, net_col, idx):
    # Маскувальна сітка = нахилена площина над/перед технікою.
    net = bproc.object.create_primitive("PLANE", scale=[s.radius_m, s.radius_m, 1.0])
    net.set_location([s.x, s.y, s.height_m])
    net.set_rotation_euler([math.radians(70.0), 0.0, s.z_rot_rad])
    net.set_cp("category_id", OCCLUDER_CATEGORY_ID)
    net.replace_materials(_mat(bproc, f"occ_net_{idx}", net_col, rough=0.8))
    return net
