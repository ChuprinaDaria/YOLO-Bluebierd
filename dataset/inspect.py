"""Швидкий аудит per-class YOLO-export датасета.

Перевіряє: структуру, лічильник кадрів і labels, орієнтовну валідність bbox
(нормалізованих 0..1), розподіл per-class, які label-файли порожні
(hard negatives).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _validate_label(path: Path) -> tuple[int, list[str]]:
    """Повертає (n_boxes, list_of_errors). n_boxes == 0 — hard negative."""
    errors: list[str] = []
    n_boxes = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_no} не 5 полів: {line!r}")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_no} нечислові поля: {line!r}")
            continue
        if cls < 0:
            errors.append(f"{path.name}:{line_no} cls<0: {cls}")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            errors.append(
                f"{path.name}:{line_no} bbox поза [0,1]: {xc},{yc},{w},{h}"
            )
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{line_no} w/h<=0: {w},{h}")
        n_boxes += 1
    return n_boxes, errors


def audit(root: Path) -> int:
    print(f"=== {root} ===")
    yaml_path = next(root.glob("data.yaml"), None)
    if yaml_path:
        meta = _read_yaml(yaml_path)
        print(f"data.yaml: nc={meta.get('nc')}, names={meta.get('names')}")
    else:
        print("data.yaml: НЕМАЄ")

    splits = {}
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        images = sorted({p.stem for p in img_dir.glob("*.jpg")})
        labels = sorted({p.stem for p in lbl_dir.glob("*.txt")}) if lbl_dir.exists() else []
        missing_lbl = set(images) - set(labels)
        orphan_lbl = set(labels) - set(images)
        cls_counter: Counter[int] = Counter()
        empty_lbls = 0
        all_errors: list[str] = []
        for lbl in (lbl_dir.glob("*.txt") if lbl_dir.exists() else []):
            n, errs = _validate_label(lbl)
            all_errors.extend(errs)
            if n == 0:
                empty_lbls += 1
            else:
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5 and parts[0].lstrip("-").isdigit():
                        cls_counter[int(parts[0])] += 1
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_lbl": len(missing_lbl),
            "orphan_lbl": len(orphan_lbl),
            "hard_negatives": empty_lbls,
            "per_class_boxes": dict(cls_counter),
            "errors": all_errors[:10],
        }

    for split, info in splits.items():
        print(f"\n[{split}]")
        for k, v in info.items():
            if k == "errors" and v:
                print(f"  errors (перші 10):")
                for e in v:
                    print(f"    - {e}")
            else:
                print(f"  {k}: {v}")
    return 0 if all(not s["errors"] for s in splits.values()) else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="root YOLO-export датасета")
    args = p.parse_args()
    if not args.path.is_dir():
        print(f"не знайдено: {args.path}", file=sys.stderr)
        return 2
    return audit(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
