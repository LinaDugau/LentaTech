"""День 2: sanity-check трекинга против GT."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
TRACKS = ROOT / "ml" / "output" / "tracks"

IOU_THR = 0.3
TS_TOL_MS = 1500


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(gt_csv: Path, pred_csv: Path) -> dict:
    gt = pd.read_csv(gt_csv)
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        gt[c] = gt[c].map(to_float)
    pred = pd.read_csv(pred_csv)

    tp = 0
    matched_gt = set()
    matched_pred = set()
    for pi, p in pred.iterrows():
        pb = (p.x_min_orig, p.y_min_orig, p.x_max_orig, p.y_max_orig)
        pts = p.frame_ts_ms
        best_iou = 0.0
        best_j = None
        for j, g in gt.iterrows():
            if j in matched_gt:
                continue
            if abs(g.frame_timestamp - pts) > TS_TOL_MS:
                continue
            gb = (g.x_min, g.y_min, g.x_max, g.y_max)
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j is not None and best_iou >= IOU_THR:
            tp += 1
            matched_gt.add(best_j)
            matched_pred.add(pi)
    P = tp / max(len(pred), 1)
    R = tp / max(len(gt), 1)
    return {"gt": len(gt), "pred": len(pred), "tp": tp,
            "precision": round(P, 3), "recall": round(R, 3)}


def main():
    for sub in ("25_12-20", "26_12-20", "43_15"):
        gt = DATA / sub / f"{sub}.csv"
        pred = TRACKS / f"track_summary_{sub}.csv"
        if not pred.exists():
            print(f"  skip {sub}: no predictions")
            continue
        m = evaluate(gt, pred)
        print(f"{sub}: gt={m['gt']:>3}  pred={m['pred']:>3}  tp={m['tp']:>3}  "
              f"P={m['precision']}  R={m['recall']}")


if __name__ == "__main__":
    sys.exit(main())
