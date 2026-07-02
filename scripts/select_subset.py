"""Select stratified subset frames for Qwen фон-pass test.

Default: 7 close (100-300m) + 7 mid (300-700m) + 6 far (700-1100m).
Балансує по seasons всередині bin. Тільки frames з non-empty label.

Usage:
    python scripts/select_subset.py <dataset_root> <out_dir> [--labels labels_min5] [--n 20]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


BINS = {"close": (100, 300), "mid": (300, 700), "far": (700, 1100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--labels", default="labels_min5")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    quotas = {"close": 7, "mid": 7, "far": args.n - 14}
    labels_dir = args.root / args.labels
    meta_dir = args.root / "metadata"

    by_bin = defaultdict(list)
    for md_path in sorted(meta_dir.glob("*.json")):
        stem = md_path.stem
        lbl = labels_dir / f"{stem}.txt"
        if not (lbl.exists() and lbl.read_text().strip()):
            continue
        data = json.loads(md_path.read_text())
        d = data["distance_m"]
        for k, (lo, hi) in BINS.items():
            if lo <= d < hi:
                by_bin[k].append((stem, d, data["season"], data["landscape"]))
                break

    random.seed(args.seed)
    picks = []
    for bin_name, quota in quotas.items():
        cand = by_bin[bin_name]
        by_season = defaultdict(list)
        for item in cand:
            by_season[item[2]].append(item)
        seasons = list(by_season.keys())
        per_season = quota // len(seasons)
        leftover = quota - per_season * len(seasons)
        for s in seasons:
            random.shuffle(by_season[s])
            picks.extend(by_season[s][:per_season])
        if leftover:
            remaining = []
            for s in seasons:
                remaining.extend(by_season[s][per_season:])
            random.shuffle(remaining)
            picks.extend(remaining[:leftover])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "picks.txt"
    out_file.write_text("\n".join(p[0] for p in picks) + "\n")

    print(f"picks: {len(picks)} written to {out_file}")
    print(f"  bin    | stem               | distance | season    | landscape")
    by_pick_bin = defaultdict(int)
    by_pick_season = defaultdict(int)
    for stem, d, s, ls in picks:
        bn = next(k for k, (lo, hi) in BINS.items() if lo <= d < hi)
        by_pick_bin[bn] += 1
        by_pick_season[s] += 1
        print(f"  {bn:<6} | {stem:<18} | {d:>6}m  | {s:<9} | {ls}")

    print(f"\n  per_bin: {dict(by_pick_bin)}")
    print(f"  per_season: {dict(by_pick_season)}")


if __name__ == "__main__":
    main()
