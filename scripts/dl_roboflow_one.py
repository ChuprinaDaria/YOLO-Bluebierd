"""Качає один Roboflow датасет за (workspace, project) у data/raw/_sources/.

Використання:
    python scripts/dl_roboflow_one.py <workspace> <project>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("workspace")
    p.add_argument("project")
    p.add_argument("--version", type=int, default=None, help="default = latest")
    p.add_argument("--format", default="yolov8")
    args = p.parse_args()

    key = os.environ.get("ROBOFLOW_WORKSPACE_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("no roboflow key", file=sys.stderr)
        return 2

    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    ws = rf.workspace(args.workspace)
    proj = ws.project(args.project)
    versions = proj.versions()
    if not versions:
        print("no versions", file=sys.stderr)
        return 3
    v = versions[-1] if args.version is None else next(
        (x for x in versions if x.version == args.version), None
    )
    if v is None:
        print(f"version {args.version} not found", file=sys.stderr)
        return 4

    out_root = REPO / "data" / "raw" / "_sources"
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / f"{args.workspace}__{args.project}__v{v.version}"
    if dest.exists() and any(dest.iterdir()):
        print(f"already exists: {dest}", file=sys.stderr)
        return 0

    print(f"downloading {args.workspace}/{args.project} v{v.version} -> {dest}")
    v.download(args.format, location=str(dest))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
