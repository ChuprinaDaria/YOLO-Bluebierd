"""Заливає одну per-class папку у приватний HF dataset repo.

Очікує що локальна папка має структуру YOLO-export:
  train/{images,labels}/  valid/{images,labels}/  test/{images,labels}/  data.yaml

Файли лягають у `<class-name>/` усередині repo. Приклад:
  python dataset/upload_class.py /tmp/yolo_recover/synthetic_apc_726/synthetic_apc_726 \\
      --class-name apc --repo Dariachup/yolo-bluebierd-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("local_folder", type=Path, help="локальна per-class YOLO-export папка")
    p.add_argument("--class-name", required=True, help="назва класу (= папка в repo)")
    p.add_argument(
        "--repo",
        default="Dariachup/yolo-bluebierd-data",
        help="приватний HF dataset repo",
    )
    p.add_argument(
        "--no-create",
        action="store_true",
        help="не створювати repo якщо не існує",
    )
    args = p.parse_args()

    if not args.local_folder.is_dir():
        print(f"не папка: {args.local_folder}", file=sys.stderr)
        return 2

    api = HfApi()
    if not args.no_create:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        print(f"repo ok: {args.repo} (private)")

    print(f"uploading {args.local_folder} -> {args.repo}:/{args.class_name}/")
    api.upload_large_folder(
        folder_path=str(args.local_folder),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.class_name,
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
