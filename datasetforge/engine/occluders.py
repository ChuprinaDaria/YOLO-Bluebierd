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
    gap_m: tuple[float, float] = (1.5, 6.0),
    lateral_m: tuple[float, float] = (-3.0, 3.0),
    tree_h: tuple[float, float] = (4.0, 9.0),
    bush_h: tuple[float, float] = (1.0, 2.5),
    net_h: tuple[float, float] = (2.0, 3.5),
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
            r = float(rng.uniform(0.8, 2.2))
        elif kind == "net":
            h = float(rng.uniform(*net_h))
            r = float(rng.uniform(2.5, 4.5))
        else:  # bush
            h = float(rng.uniform(*bush_h))
            r = float(rng.uniform(0.8, 1.8))
        specs.append(OccluderSpec(
            kind=kind, x=ox, y=oy, z=0.0,
            height_m=h, radius_m=r,
            z_rot_rad=float(rng.uniform(0, 2 * math.pi)),
        ))
    return specs


def build_occluders(specs: list[OccluderSpec], season: str):
    """Побудувати 3D-примітиви оклюдерів у сцені. Lazy bproc/bpy.

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

    objs = []
    for i, s in enumerate(specs):
        try:
            if s.kind == "tree":
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


def _mat(bproc, name, color, rough=0.9):
    m = bproc.material.create(name)
    m.set_principled_shader_value("Base Color", color)
    m.set_principled_shader_value("Roughness", rough)
    m.set_principled_shader_value("Specular", 0.05)
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
