"""Анализ пар 17–22/23 полей."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "evaluate_official", ROOT / "ml" / "scripts" / "evaluate_official.py"
)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)

ALL = _ev.ALL_CONTENT_FIELDS
match_pairs = _ev.match_pairs
field_match = _ev.field_match
DATA = _ev.DATA


def analyze(video: str, pred_dir: Path, min_hits: int = 17, max_hits: int = 22) -> None:
    gt_path = DATA / video / f"{video}.csv"
    pred_path = pred_dir / f"final_eval_{video}.csv"
    if not pred_path.is_file():
        pred_path = pred_dir / f"final_{video}.csv"
    gt = pd.read_csv(gt_path).rename(columns=_ev.GT_COLUMN_RENAME)
    pred = pd.read_csv(pred_path)
    pairs = match_pairs(gt, pred)

    print(f"\n=== {video} near-miss ({min_hits}–{max_hits}/23) ===")
    n = 0
    for pi, gi, mt in pairs:
        gt_row = gt.loc[gi]
        pr_row = pred.loc[pi]
        hits = []
        misses = []
        for f in ALL:
            if f not in gt_row.index or f not in pr_row.index:
                misses.append(f)
                continue
            if field_match(f, gt_row[f], pr_row[f]):
                hits.append(f)
            else:
                misses.append(f)
        nh = len(hits)
        if min_hits <= nh <= max_hits:
            n += 1
            bc = str(pr_row.get("barcode", ""))[:13]
            print(f"  pair {n}: {nh}/23 [{mt}] bc={bc!r} miss={misses[:8]}{'…' if len(misses)>8 else ''}")
    if not n:
        print("  (none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "ml" / "output")
    ap.add_argument("--min-hits", type=int, default=17)
    ap.add_argument("--max-hits", type=int, default=22)
    ap.add_argument("--video", action="append")
    args = ap.parse_args()
    videos = args.video or ["25_12-20", "26_12-20", "43_15"]
    for v in videos:
        analyze(v, args.pred_dir, args.min_hits, args.max_hits)


if __name__ == "__main__":
    main()
