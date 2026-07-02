"""G2 gate: KS-тест bbox-distribution нашого ifv_apc проти style anchor _synthetic_apc_726.

Не вимагає Blender. Читає YOLO labels з двох директорій, рахує bbox short-side
у пікселях, проганяє Kolmogorov-Smirnov. Pass: median 60 px ± 15 px AND KS p > 0.05.

Локально, без даних анкору, тест skipped через pytest.skip — gating відбувається
у Colab notebook де датасет доступний.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


ANCHOR_DEFAULT = Path("data/raw/_synthetic_apc_726")


def _read_yolo_short_sides_px(labels_dir: Path, image_w: int, image_h: int) -> list[float]:
    """Повертає список min(box_w_px, box_h_px) для всіх bbox у директорії."""
    sides: list[float] = []
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _cls, _xc, _yc, bw, bh = (float(p) for p in parts)
            sides.append(min(bw * image_w, bh * image_h))
    return sides


@pytest.mark.skipif(
    not (ANCHOR_DEFAULT / "train" / "labels").exists(),
    reason="anchor dataset not available locally",
)
def test_anchor_distribution_median_within_target():
    """Перевірка що сам anchor (`_synthetic_apc_726`) має median 60 px ± 15 (sanity для тесту)."""
    import statistics
    sides = _read_yolo_short_sides_px(
        ANCHOR_DEFAULT / "train" / "labels",
        image_w=640, image_h=640,
    )
    assert len(sides) > 0
    med = statistics.median(sides)
    assert 45 <= med <= 75, f"anchor median {med:.1f}px out of expected range"


def test_ks_helper_imports():
    """Sanity: scipy.stats доступний для KS тесту (запуск у Colab/HF Jobs)."""
    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        pytest.skip("scipy not installed locally — test active у Colab/HF Jobs")


@pytest.mark.skipif(
    "GENERATED_LABELS_DIR" not in os.environ,
    reason="run with GENERATED_LABELS_DIR=/path/to/labels у Colab post-Phase 1.5",
)
def test_generated_matches_anchor_ks():
    """G2 gate: бенчмарк згенерованих ifv_apc labels проти anchor."""
    import statistics
    from scipy import stats

    gen_dir = Path(os.environ["GENERATED_LABELS_DIR"])
    anchor_dir = Path(os.environ.get("ANCHOR_LABELS_DIR", ANCHOR_DEFAULT / "train" / "labels"))

    gen_sides = _read_yolo_short_sides_px(gen_dir, image_w=640, image_h=640)
    anchor_sides = _read_yolo_short_sides_px(anchor_dir, image_w=640, image_h=640)
    assert gen_sides and anchor_sides

    gen_med = statistics.median(gen_sides)
    assert 45 <= gen_med <= 75, f"generated median {gen_med:.1f}px out of 60±15 target"

    ks_stat, p_value = stats.ks_2samp(gen_sides, anchor_sides)
    assert p_value > 0.05, f"KS p={p_value:.4f} < 0.05 — distributions diverge (stat={ks_stat:.3f})"
