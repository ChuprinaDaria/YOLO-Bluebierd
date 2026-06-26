"""Season → HDRI + ground texture path resolver.

Не торкається bpy, просто матчить season name до доступних assets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SeasonAssets:
    season: str
    hdri: Path
    ground_texture: Path


def pick_season_assets(
    assets_root: Path,
    season: str,
    seed: int = 0,
) -> SeasonAssets:
    """Випадково (з seed) обирає одну HDRI + одну ground texture для season."""
    hdri_dir = assets_root / "hdri" / season
    ground_dir = assets_root / "textures" / "ground" / season

    hdris = sorted(hdri_dir.glob("*.hdr")) + sorted(hdri_dir.glob("*.exr"))
    grounds = sorted(ground_dir.glob("*_diff_*.png")) + sorted(ground_dir.glob("*_diff_*.jpg"))

    if not hdris:
        raise FileNotFoundError(f"no HDRIs in {hdri_dir}")
    if not grounds:
        raise FileNotFoundError(f"no ground textures in {ground_dir}")

    rng = random.Random(seed)
    return SeasonAssets(
        season=season,
        hdri=rng.choice(hdris),
        ground_texture=rng.choice(grounds),
    )
