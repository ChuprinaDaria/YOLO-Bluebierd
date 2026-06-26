"""datasetforge.engine — Blender 3D composite render engine.

Запускається через BlenderProc:
    blenderproc run datasetforge/engine/render_runner.py --config ... --n N --out OUT

Локально модулі importable для linting/тестів (bpy/bproc — lazy import всередині build_scene).
"""

__version__ = "0.2.0-dev"
