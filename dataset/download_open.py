"""Качає Roboflow + Kaggle датасети за `dataset/sources.yaml` у `data/raw/_sources/`.

Тільки CC BY 4.0 / MIT (сумісні з defense commercial).
Кожен датасет лягає в окрему папку `<workspace>__<project>__v<v>/` для Roboflow і
`kaggle/<ref-slug>/` для Kaggle.

Використання:
    python dataset/download_open.py                  # всі priority<=2
    python dataset/download_open.py --priority 1     # тільки топ
    python dataset/download_open.py --only roboflow  # фільтр джерела
    python dataset/download_open.py --dry-run        # без скачування
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

OUT = REPO / "data" / "raw" / "_sources"
OUT.mkdir(parents=True, exist_ok=True)


def download_roboflow(items: list[dict], dry: bool) -> None:
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_WORKSPACE_KEY") or os.environ.get(
        "ROBOFLOW_API_KEY"
    )
    if not key:
        print("[roboflow] no key", file=sys.stderr)
        return
    rf = Roboflow(api_key=key)
    for item in items:
        ws_id = item["workspace"]
        proj_id = item["project"]
        dest_stub = f"{ws_id}__{proj_id}"
        existing = sorted(OUT.glob(f"{dest_stub}__v*"))
        if existing:
            print(f"[roboflow] {dest_stub}: вже є {existing[-1].name}, скіп")
            continue
        if dry:
            print(f"[roboflow] DRY {dest_stub}")
            continue
        try:
            ws = rf.workspace(ws_id)
            proj = ws.project(proj_id)
            versions = proj.versions()
            if not versions:
                print(f"[roboflow] {dest_stub}: версій немає, скіп")
                continue
            v = versions[-1]
            v_num = v.version
            dest = OUT / f"{dest_stub}__v{v_num}"
            print(f"[roboflow] {dest_stub} v{v_num} -> {dest.name}")
            v.download("yolov8", location=str(dest))
        except Exception as e:
            print(f"[roboflow] {dest_stub}: FAIL — {type(e).__name__}: {e}")


def download_kaggle(items: list[dict], dry: bool) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"[kaggle] auth FAIL: {e}", file=sys.stderr)
        return
    kdir = OUT / "kaggle"
    kdir.mkdir(exist_ok=True)
    for item in items:
        ref = item["ref"]
        slug = ref.replace("/", "__")
        dest = kdir / slug
        if dest.exists() and any(dest.iterdir()):
            print(f"[kaggle] {ref}: вже є, скіп")
            continue
        if dry:
            print(f"[kaggle] DRY {ref}")
            continue
        dest.mkdir(exist_ok=True)
        print(f"[kaggle] {ref} -> {dest.name}")
        try:
            api.dataset_download_files(ref, path=str(dest), unzip=True)
        except Exception as e:
            print(f"[kaggle] {ref}: FAIL — {type(e).__name__}: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "dataset" / "sources.yaml"))
    p.add_argument("--priority", type=int, default=2, help="завантажувати з priority<=N")
    p.add_argument("--only", choices=["roboflow", "kaggle"], help="фільтр джерела")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rb = [x for x in (cfg.get("roboflow") or []) if x.get("priority", 99) <= args.priority]
    kg = [x for x in (cfg.get("kaggle") or []) if x.get("priority", 99) <= args.priority]

    print(f"plan: roboflow={len(rb)}, kaggle={len(kg)}, priority<={args.priority}")
    if not args.only or args.only == "roboflow":
        download_roboflow(rb, args.dry_run)
    if not args.only or args.only == "kaggle":
        download_kaggle(kg, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
