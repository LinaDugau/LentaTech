"""День 11: официальная метрика организаторов."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
OUT = ROOT / "ml" / "output"

IOU_THR = 0.20
TS_TOL_MS = 3000

TAG_FIELDS = [
    "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime",
    "code", "additional_info", "color", "special_symbols",
]
QR_FIELDS = [
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]
ALL_CONTENT_FIELDS = TAG_FIELDS + QR_FIELDS  # 23

PRICE_FIELDS = {
    "price_default", "price_card", "price_discount",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_price", "wholesale_level_2_price",
    "action_price_qr",
}
INTEGER_FIELDS = {
    "wholesale_level_1_count", "wholesale_level_2_count",
}
SUCCESS_THRESHOLD = 0.80


def _norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def _is_no(x: str) -> bool:
    return _norm_str(x).lower() in ("нет", "no", "n/a", "n\\a")


def _is_empty(x: str) -> bool:
    return _norm_str(x) == ""


def _to_float(x):
    s = _norm_str(x).replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_str_canonical(x: str) -> str:
    s = _norm_str(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _toks4(s: str) -> set[str]:
    return set(t for t in re.findall(r"[a-zа-я0-9]+", _norm_str(s).lower())
               if len(t) >= 4)


def field_match(field: str, gt_val, pred_val) -> bool:
    """True если pred-значение совпадает с GT по правилам поля.

    Семантика:
    - «нет» в GT означает «поле не предусмотрено на этом ценнике»
    - пусто в GT — поле просто не размечено (организаторская халтура);
      не штрафуем pred за любой ответ.
    """
    gt_no = _is_no(gt_val)
    pr_no = _is_no(pred_val)
    gt_empty = _is_empty(gt_val) and not gt_no
    pr_empty = _is_empty(pred_val) and not pr_no

    if gt_empty:
        return True
    if gt_no and pr_no:
        return True
    if gt_no and not pr_no:
        return False
    if not gt_no and (pr_no or pr_empty):
        return False

    if field in PRICE_FIELDS:
        fa, fb = _to_float(gt_val), _to_float(pred_val)
        if fa is None or fb is None:
            return False
        if fa == fb:
            return True
        if abs(fa - fb) / max(fa, 1e-9) <= 0.01:
            return True
        if int(fa) == int(fb):
            return True
        return False

    if field in INTEGER_FIELDS:
        fa, fb = _to_float(gt_val), _to_float(pred_val)
        return fa is not None and fb is not None and int(fa) == int(fb)

    if field == "product_name":
        ta, tb = _toks4(gt_val), _toks4(pred_val)
        return bool(ta and tb and (ta & tb))

    if field == "barcode" or field == "qr_code_barcode":
        def _ean(x):
            s = _norm_str(x)
            if s.endswith(".0"):
                s = s[:-2]
            if "e" in s.lower() and re.match(r"^-?\d", s):
                try:
                    s = f"{int(float(s))}"
                except Exception:
                    pass
            return re.sub(r"\D", "", s)
        ga, gb = _ean(gt_val), _ean(pred_val)
        return ga == gb and ga != ""

    if field == "id_sku":
        def _idnorm(x):
            s = _norm_str(x)
            if s.endswith(".0"):
                s = s[:-2]
            return re.sub(r"\D", "", s)
        ga, gb = _idnorm(gt_val), _idnorm(pred_val)
        return ga == gb and ga != ""

    return _norm_str_canonical(gt_val) == _norm_str_canonical(pred_val)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _digits_only(x) -> str:
    s = _norm_str(x)
    if s.endswith(".0"):
        s = s[:-2]
    if "e" in s.lower() and re.match(r"^-?\d", s):
        try:
            s = f"{int(float(s))}"
        except Exception:
            pass
    return re.sub(r"\D", "", s)


def match_pairs(gt: pd.DataFrame, pred: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Возвращает [(pred_idx, gt_idx, match_type)].

    Шаги:
    1) barcode-match (приоритет №1): EAN-13 совпал → пара.
    2) для оставшихся — spatio-temporal с alt-bbox если есть alts_json.
    """
    import json as _json
    matched_gt = set()
    matched_pred = set()
    pairs: list[tuple[int, int, str]] = []

    gt = gt.copy()
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        nm = f"_{c}"
        gt[nm] = gt[c].map(_to_float)
    gt["_bc"] = gt["barcode"].map(_digits_only)

    pred = pred.copy()
    pred["_bc"] = pred["barcode"].map(_digits_only)

    for pi, p in pred.iterrows():
        bc = p._bc
        if not re.fullmatch(r"\d{12,13}", bc):
            continue
        cand = gt[(gt._bc == bc) & (~gt.index.isin(matched_gt))]
        if len(cand) == 1:
            gi = cand.index[0]
            pairs.append((pi, gi, "barcode"))
            matched_gt.add(gi)
            matched_pred.add(pi)

    pred_alts: dict[int, list[tuple[float, tuple]]] = {}
    for pi, p in pred.iterrows():
        if pi in matched_pred:
            continue
        cand_boxes = [(p.frame_timestamp,
                       (p.x_min, p.y_min, p.x_max, p.y_max))]
        aj = p.get("alts_json", "") if "alts_json" in pred.columns else ""
        if isinstance(aj, str) and aj.strip():
            try:
                for a in _json.loads(aj):
                    cand_boxes.append((a["ts_ms"],
                                       (a["x_min"], a["y_min"],
                                        a["x_max"], a["y_max"])))
            except Exception:
                pass
        pred_alts[pi] = cand_boxes

    pi_order = sorted(pred_alts.keys(),
                      key=lambda i: -((pred.loc[i, "x_max"] - pred.loc[i, "x_min"])
                                       * (pred.loc[i, "y_max"] - pred.loc[i, "y_min"])))
    for pi in pi_order:
        best_iou = 0.0
        best = None
        for gi, g in gt.iterrows():
            if gi in matched_gt:
                continue
            for pts, pb in pred_alts[pi]:
                if abs(g._frame_timestamp - pts) > TS_TOL_MS:
                    continue
                v = iou(pb, (g._x_min, g._y_min, g._x_max, g._y_max))
                if v > best_iou:
                    best_iou = v
                    best = gi
        if best is not None and best_iou >= IOU_THR:
            pairs.append((pi, best, "spatio"))
            matched_gt.add(best)
            matched_pred.add(pi)

    return pairs


GT_COLUMN_RENAME = {
    "wholesale_level_1_coun": "wholesale_level_1_count",
    "wholesale_level_2_coun": "wholesale_level_2_count",
}


def evaluate(video: str):
    gt_path = DATA / video / f"{video}.csv"
    pred_path = OUT / f"final_eval_{video}.csv"
    if not pred_path.exists():
        pred_path = OUT / f"final_{video}.csv"
    gt = pd.read_csv(gt_path).rename(columns=GT_COLUMN_RENAME)
    pred = pd.read_csv(pred_path)

    pairs = match_pairs(gt, pred)

    per_field_hits = {f: 0 for f in ALL_CONTENT_FIELDS}
    per_field_total = {f: 0 for f in ALL_CONTENT_FIELDS}
    success_count = 0
    per_pair_scores: list[float] = []
    by_match_type = {"barcode": 0, "spatio": 0}

    for pi, gi, mt in pairs:
        gt_row = gt.loc[gi]
        pr_row = pred.loc[pi]
        hits = 0
        for f in ALL_CONTENT_FIELDS:
            if f not in gt_row.index or f not in pr_row.index:
                per_field_total[f] += 1
                continue
            ok = field_match(f, gt_row[f], pr_row[f])
            per_field_total[f] += 1
            if ok:
                per_field_hits[f] += 1
                hits += 1
        score = hits / len(ALL_CONTENT_FIELDS)
        per_pair_scores.append(score)
        if score >= SUCCESS_THRESHOLD:
            success_count += 1
            by_match_type[mt] += 1

    n_gt = len(gt)
    n_pred = len(pred)
    n_matched = len(pairs)
    target_metric = success_count / max(n_gt, 1)

    print(f"\n=== {video} ===")
    print(f"  GT={n_gt}  Pred={n_pred}  matched={n_matched}  "
          f"(barcode={sum(1 for _,_,mt in pairs if mt=='barcode')}, "
          f"spatio={sum(1 for _,_,mt in pairs if mt=='spatio')})")
    if n_matched:
        avg_score = sum(per_pair_scores) / n_matched
        print(f"  avg per-pair score (на сматченных): {avg_score:.3f}")
        print(f"  ≥80% per-pair: {success_count}/{n_matched} матчей")
    print(f"  TARGET_METRIC = {success_count}/{n_gt} = {target_metric:.3f}")
    print(f"  per-field hits (на {n_matched} сматченных):")
    for f in ALL_CONTENT_FIELDS:
        h = per_field_hits[f]
        t = per_field_total[f]
        pct = h / t if t else 0
        print(f"    {f:<28} {h:>3}/{t:<3}  {pct*100:5.1f}%")
    return target_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*",
                    default=["25_12-20", "26_12-20", "43_15"])
    args = ap.parse_args()
    overall_n_gt = 0
    overall_success = 0
    for v in args.videos:
        gt_path = DATA / v / f"{v}.csv"
        if not gt_path.exists():
            continue
        gt = pd.read_csv(gt_path).rename(columns=GT_COLUMN_RENAME)
        pred_path = OUT / f"final_eval_{v}.csv"
        if not pred_path.exists():
            pred_path = OUT / f"final_{v}.csv"
        if not pred_path.exists():
            print(f"skip {v}: no pred")
            continue
        pred = pd.read_csv(pred_path)
        pairs = match_pairs(gt, pred)
        cnt = 0
        for pi, gi, _ in pairs:
            gt_row = gt.loc[gi]; pr_row = pred.loc[pi]
            hits = sum(field_match(f, gt_row.get(f), pr_row.get(f))
                        for f in ALL_CONTENT_FIELDS
                        if f in gt_row.index and f in pr_row.index)
            if hits / len(ALL_CONTENT_FIELDS) >= SUCCESS_THRESHOLD:
                cnt += 1
        overall_success += cnt
        overall_n_gt += len(gt)
        evaluate(v)

    print(f"\n=== ИТОГО ===")
    print(f"  TARGET_METRIC (overall) = {overall_success}/{overall_n_gt} = "
          f"{overall_success / max(overall_n_gt, 1):.3f}")


if __name__ == "__main__":
    sys.exit(main())
