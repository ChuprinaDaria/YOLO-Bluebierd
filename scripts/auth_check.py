"""Перевіряє auth до Roboflow і Kaggle без echo секретів у shell.

Зчитує .env через python-dotenv. Виводить лише прапорці OK/FAIL.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def check_roboflow() -> None:
    candidates = [
        ("ROBOFLOW_API_KEY", os.environ.get("ROBOFLOW_API_KEY")),
        ("ROBOFLOW_WORKSPACE_KEY", os.environ.get("ROBOFLOW_WORKSPACE_KEY")),
    ]
    from roboflow import Roboflow

    for name, key in candidates:
        if not key:
            print(f"ROBOFLOW[{name}]: no key")
            continue
        try:
            rf = Roboflow(api_key=key)
            ws = rf.workspace()
            print(f"ROBOFLOW[{name}]: OK ({ws.url})")
            return
        except Exception as e:
            msg = str(e)[:120]
            print(f"ROBOFLOW[{name}]: FAIL — {msg}")


def check_kaggle() -> None:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("KAGGLE: no token")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        result = api.dataset_list(search="rawsi18", page=1)
        print(f"KAGGLE: OK (sample query returned {len(result)} datasets)")
    except Exception as e:
        print(f"KAGGLE: FAIL — {type(e).__name__}: {e}")


if __name__ == "__main__":
    check_roboflow()
    check_kaggle()
