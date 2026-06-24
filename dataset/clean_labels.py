"""Очищує/конвертує YOLO-label файли в одному форматі (detection bbox).

Знаходить:
- Polygon segmentation рядки (>5 полів) → конвертує в axis-aligned bbox
- Бокси з w<=0 або h<=0 → видаляє
- Не-числовий class_id (наприклад `3.0`) → нормалізує до int
- Рядки з некоректним числом полів → видаляє

Перезаписує файли in-place (за замовч.) або в `--dest`.
Логує статистику у stderr.

Використання:
    python dataset/clean_labels.py <labels_dir>
    python dataset/clean_labels.py <labels_dir> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_bbox(parts: list[str]) -> tuple[int, float, float, float, float] | None:
    """Парсить YOLO detection bbox: class xc yc w h."""
    try:
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(x) for x in parts[1:5])
    except (ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    return cls, xc, yc, w, h


def polygon_to_bbox(parts: list[str]) -> tuple[int, float, float, float, float] | None:
    """Конвертує YOLO polygon (class x1 y1 x2 y2 ...) → axis-aligned bbox."""
    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except (ValueError, IndexError):
        return None
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    xc = x_min + w / 2
    yc = y_min + h / 2
    return cls, xc, yc, w, h


def clean_file(path: Path) -> tuple[str, dict[str, int]]:
    """Повертає (cleaned_text, stats)."""
    stats = {"kept_bbox": 0, "converted_poly": 0, "dropped_invalid": 0, "dropped_zero_wh": 0}
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        n = len(parts)
        if n == 5:
            res = parse_bbox(parts)
            if res is None:
                if parts[0].replace(".", "").lstrip("-").isdigit():
                    stats["dropped_zero_wh"] += 1
                else:
                    stats["dropped_invalid"] += 1
                continue
            stats["kept_bbox"] += 1
        elif n >= 9 and (n - 1) % 2 == 0:
            res = polygon_to_bbox(parts)
            if res is None:
                stats["dropped_invalid"] += 1
                continue
            stats["converted_poly"] += 1
        else:
            stats["dropped_invalid"] += 1
            continue
        cls, xc, yc, w, h = res
        out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out_lines) + ("\n" if out_lines else ""), stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("labels_dir", type=Path)
    p.add_argument("--dest", type=Path, help="куди писати (за замовч. in-place)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.labels_dir.is_dir():
        print(f"not a dir: {args.labels_dir}", file=sys.stderr)
        return 2

    total = {"files": 0, "kept_bbox": 0, "converted_poly": 0, "dropped_invalid": 0, "dropped_zero_wh": 0, "emptied": 0}
    for f in sorted(args.labels_dir.rglob("*.txt")):
        rel = f.relative_to(args.labels_dir)
        cleaned, stats = clean_file(f)
        total["files"] += 1
        for k, v in stats.items():
            total[k] += v
        if not cleaned.strip():
            total["emptied"] += 1
        if args.dry_run:
            continue
        out_path = args.dest / rel if args.dest else f
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")

    print("=== summary ===", file=sys.stderr)
    for k, v in total.items():
        print(f"  {k}: {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
