"""Стягнути безкоштовні HDRI + ground-текстури з Poly Haven по сезонах.

Заповнює `assets/hdri/<season>/` і `assets/textures/ground/<season>/` під
`render_runner` (див. datasetforge/assets/README.md). Моделі техніки —
вручну (ліцензії/пошук окремо), скрипт їх НЕ качає.

Тільки stdlib (urllib/json) — без залежностей. Poly Haven CC0, атрибуція не
обов'язкова. Запускати там, де є мережа (локаль/RunPod, НЕ у цьому sandbox).

Приклади:
  python datasetforge/tools/fetch_assets.py --assets-root datasetforge/assets
  python datasetforge/tools/fetch_assets.py --res 4k --seasons summer winter
  python datasetforge/tools/fetch_assets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"

# Сезон → ключові слова для матчингу тегів/категорій/назви Poly Haven-ассета.
# Перший матч перемагає; якщо жоден не підійшов — беремо перший outdoor-ассет.
SEASON_HINTS = {
    "summer":     {"hdri": ["sunny", "field", "meadow", "grass", "day"],
                   "ground": ["grass", "meadow", "field", "lawn"]},
    "autumn_mud": {"hdri": ["overcast", "cloudy", "autumn", "field"],
                   "ground": ["mud", "dirt", "soil", "ground", "forest"]},
    "winter":     {"hdri": ["snow", "winter", "overcast", "cloudy"],
                   "ground": ["snow", "ice"]},
    "spring":     {"hdri": ["field", "meadow", "partly", "day", "sunny"],
                   "ground": ["field", "grass", "dirt", "soil"]},
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "datasetforge/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "datasetforge/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _score(meta: dict, keywords: list[str]) -> int:
    hay = " ".join(str(x).lower() for x in
                   (meta.get("tags", []) + meta.get("categories", [])
                    + [meta.get("name", "")]))
    return sum(1 for kw in keywords if kw in hay)


def _pick_slug(assets: dict, keywords: list[str]) -> str | None:
    ranked = sorted(assets.items(),
                    key=lambda kv: _score(kv[1], keywords), reverse=True)
    for slug, meta in ranked:
        if _score(meta, keywords) > 0:
            return slug
    # fallback: перший будь-який (outdoor за замовч. для hdris/textures API)
    return next(iter(assets), None)


def _pick_res(res_map: dict, want: str) -> str:
    if want in res_map:
        return want
    order = ["2k", "1k", "4k", "8k"]
    for r in order:
        if r in res_map:
            return r
    return next(iter(res_map))  # хоч щось


def _hdri_url(slug: str, want_res: str):
    files = _get_json(f"{API}/files/{slug}")
    hdri = files.get("hdri", {})
    if not hdri:
        return None, None
    res = _pick_res(hdri, want_res)
    entry = hdri[res]
    fmt = "hdr" if "hdr" in entry else ("exr" if "exr" in entry else None)
    if not fmt:
        return None, None
    return entry[fmt]["url"], f"{slug}_{res}.{fmt}"


def _diffuse_url(slug: str, want_res: str):
    files = _get_json(f"{API}/files/{slug}")
    # мапа дифузу зветься по-різному: Diffuse / diffuse / albedo / col_...
    diff_key = next((k for k in files
                     if any(t in k.lower() for t in ("diff", "albedo", "col"))
                     and isinstance(files[k], dict)), None)
    if not diff_key:
        return None, None
    res_map = files[diff_key]
    res = _pick_res(res_map, want_res)
    entry = res_map[res]
    fmt = "jpg" if "jpg" in entry else ("png" if "png" in entry else None)
    if not fmt:
        return None, None
    # ІМ'Я МУСИТЬ містити _diff_ (glob у season_lighting: *_diff_*).
    return entry[fmt]["url"], f"{slug}_diff_{res}.{fmt}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", type=Path, default=Path("datasetforge/assets"))
    ap.add_argument("--seasons", nargs="+",
                    default=["summer", "autumn_mud", "winter", "spring"])
    ap.add_argument("--res", default="2k", help="2k | 4k | 1k | 8k")
    ap.add_argument("--dry-run", action="store_true",
                    help="лише показати, що б завантажилось")
    args = ap.parse_args(argv)

    print(f"[fetch] Poly Haven → {args.assets_root} (res={args.res}, "
          f"seasons={args.seasons}) {'[DRY-RUN]' if args.dry_run else ''}")
    try:
        hdris = _get_json(f"{API}/assets?t=hdris")
        textures = _get_json(f"{API}/assets?t=textures")
    except Exception as exc:
        print(f"[err] Poly Haven API недоступний ({exc.__class__.__name__}: {exc}). "
              f"Запускай там, де є мережа (не у sandbox).", file=sys.stderr)
        return 2

    ok, fail = 0, 0
    for season in args.seasons:
        hints = SEASON_HINTS.get(season, {"hdri": [], "ground": []})

        h_slug = _pick_slug(hdris, hints["hdri"])
        g_slug = _pick_slug(textures, hints["ground"])
        try:
            h_url, h_name = _hdri_url(h_slug, args.res) if h_slug else (None, None)
            g_url, g_name = _diffuse_url(g_slug, args.res) if g_slug else (None, None)
        except Exception as exc:
            print(f"[warn] {season}: files API помилка ({exc}) — пропуск")
            fail += 1
            continue

        for kind, url, name, sub in (
            ("hdri", h_url, h_name, args.assets_root / "hdri" / season),
            ("ground", g_url, g_name, args.assets_root / "textures" / "ground" / season),
        ):
            if not url:
                print(f"[warn] {season}/{kind}: не знайдено відповідного ассета")
                fail += 1
                continue
            dst = sub / name
            print(f"  {season:11s} {kind:6s} → {dst}")
            if args.dry_run:
                continue
            try:
                _download(url, dst)
                ok += 1
            except Exception as exc:
                print(f"[warn] download fail {url}: {exc}")
                fail += 1

    print(f"[done] downloaded={ok} failed/skipped={fail}. "
          f"Моделі техніки клади вручну у assets/models/<class>/ (див. README).")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
