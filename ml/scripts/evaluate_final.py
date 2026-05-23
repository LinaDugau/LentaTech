"""Eval final CSV."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
OUT = ROOT / "ml" / "output"


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None


def norm_price(x):
    import math
    f = to_float(x)
    if f is None or (isinstance(f, float) and math.isnan(f)):
        return None
    return round(f, 2)


IOU_THR = 0.2
TS_TOL_MS = 3000


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def spatio_temporal_match(gt: pd.DataFrame, pred: pd.DataFrame) -> list[tuple[int, int]]:
    """Greedy match: для каждого pred — лучший GT.
    Используем все K-rank bbox если они есть в `alts_json`.
    """
    import json
    pairs = []
    matched = set()
    pred_alts: dict[int, list[tuple[float, tuple]]] = {}
    for pi, p in pred.iterrows():
        cand = [(p.frame_timestamp, (p.x_min, p.y_min, p.x_max, p.y_max))]
        aj = p.get("alts_json", "")
        if isinstance(aj, str) and aj.strip():
            try:
                for a in json.loads(aj):
                    cand.append((a["ts_ms"],
                                 (a["x_min"], a["y_min"], a["x_max"], a["y_max"])))
            except Exception:
                pass
        pred_alts[pi] = cand

    pi_order = sorted(pred.index,
                      key=lambda i: -((pred.loc[i, "x_max"] - pred.loc[i, "x_min"])
                                       * (pred.loc[i, "y_max"] - pred.loc[i, "y_min"])))
    for pi in pi_order:
        best_iou = 0.0
        best = None
        for gi, g in gt.iterrows():
            if gi in matched:
                continue
            for pts, pb in pred_alts[pi]:
                if abs(g._ts - pts) > TS_TOL_MS:
                    continue
                v = iou(pb, (g._x1, g._y1, g._x2, g._y2))
                if v > best_iou:
                    best_iou = v
                    best = gi
        if best is not None and best_iou >= IOU_THR:
            pairs.append((pi, best))
            matched.add(best)
    return pairs


def evaluate(video: str):
    gt = pd.read_csv(DATA / video / f"{video}.csv")
    pred_path = OUT / f"final_eval_{video}.csv"
    if not pred_path.exists():
        pred_path = OUT / f"final_{video}.csv"
    pred = pd.read_csv(pred_path)
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        gt[f"_{c.replace('frame_timestamp','ts').replace('x_min','x1').replace('y_min','y1').replace('x_max','x2').replace('y_max','y2')}"] = gt[c].map(to_float)

    def norm_bc(s):
        s = str(s).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s
    gt["barcode_norm"] = gt["barcode"].map(norm_bc)
    pred["barcode_norm"] = pred["barcode"].map(norm_bc)

    print(f"\n=== {video} ===")
    print(f"  GT={len(gt)}  Pred={len(pred)}")

    pairs = spatio_temporal_match(gt, pred)
    tp = len(pairs)
    P = tp / max(len(pred), 1)
    R = tp / max(len(gt), 1)
    f1 = 2 * P * R / max(P + R, 1e-9)
    print(f"  spatio-temporal: tp={tp}  P={P:.3f}  R={R:.3f}  F1={f1:.3f}")

    fields = ("price_default", "price_card", "color", "discount_amount",
              "product_name")
    ok_counts = {f: 0 for f in fields}
    for pi, gi in pairs:
        for f in fields:
            a, b = gt.loc[gi, f], pred.loc[pi, f]
            if f.startswith("price"):
                pa, pb = norm_price(a), norm_price(b)
                if pa is not None and pb is not None:
                    if (pa == pb
                            or abs(pa - pb) / max(pa, 1e-9) <= 0.01
                            or int(pa) == int(pb)):
                        ok_counts[f] += 1
            elif f == "product_name":
                import re as _re
                def toks(s):
                    s = str(s).lower()
                    return set(t for t in _re.findall(r"[a-zа-я0-9]+", s)
                               if len(t) >= 4)
                ta, tb = toks(a), toks(b)
                if ta and tb and ta & tb:
                    ok_counts[f] += 1
            else:
                if str(a).strip().lower() == str(b).strip().lower() and str(a).strip().lower() not in ("", "nan", "нет"):
                    ok_counts[f] += 1
    print("  per-field accuracy on matched:")
    for f in fields:
        print(f"    {f}: {ok_counts[f]}/{tp}")


def main():
    for v in ("25_12-20", "26_12-20", "43_15"):
        evaluate(v)


if __name__ == "__main__":
    sys.exit(main())
