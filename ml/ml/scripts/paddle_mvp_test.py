"""MVP-тест PaddleOCR на near-miss парах 26_12-20 (h=15-18)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_official import match_pairs, field_match, ALL_CONTENT_FIELDS, GT_COLUMN_RENAME

ROOT = Path(__file__).resolve().parents[2]
VIDEO = "26_12-20"
H_RANGE = (12, 18)


def collect_targets():
    gt = pd.read_csv(f"Данные/{VIDEO}/{VIDEO}.csv").rename(columns=GT_COLUMN_RENAME)
    pred = pd.read_csv(f"ml/output/final_eval_{VIDEO}.csv", dtype={"barcode": str})
    pairs = match_pairs(gt, pred)
    rows = []
    for pi, gi, mt in pairs:
        h = sum(field_match(f, gt.loc[gi].get(f), pred.loc[pi].get(f))
                for f in ALL_CONTENT_FIELDS if f in gt.columns and f in pred.columns)
        if H_RANGE[0] <= h <= H_RANGE[1]:
            rows.append((h, mt, gi, pi))
    rows.sort(key=lambda r: -r[0])
    return gt, pred, rows


def get_hires_for_pred(pred_row, ocr_df):
    cand = ocr_df[
        (ocr_df.frame_ts_ms == pred_row.frame_timestamp)
        & (ocr_df.x_min_orig == pred_row.x_min)
        & (ocr_df.y_min_orig == pred_row.y_min)
    ]
    if cand.empty:
        return None
    cp = cand.iloc[0].get("hires_crop") or cand.iloc[0].get("crop_path")
    if not isinstance(cp, str):
        return None
    p = ROOT / cp
    return p if p.exists() else None


def main():
    gt, pred, targets = collect_targets()
    ocr_df = pd.read_csv(f"ml/output/ocr_qr_{VIDEO}.csv")

    print(f"Found {len(targets)} pairs on {VIDEO} with h={H_RANGE[0]}..{H_RANGE[1]}")
    print()

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang="ru", device="cpu")
    print("Paddle ready.\n")

    hits_total = 0
    by_field = {"id_sku": 0, "print_datetime": 0, "code": 0, "product_name": 0,
                "additional_info": 0, "discount_amount": 0,
                "price_default": 0, "price_card": 0}
    seen_field = {k: 0 for k in by_field}

    for h, mt, gi, pi in targets[:8]:
        pr = pred.iloc[pi]
        g = gt.iloc[gi]
        cp = get_hires_for_pred(pr, ocr_df)
        if cp is None:
            continue
        img = cv2.imread(str(cp))
        if img is None:
            continue
        try:
            result = ocr.predict(img)
        except Exception as e:
            print(f"  paddle error: {e}")
            continue
        res = result[0].json.get("res", {}) if result else {}
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        all_txt = " ".join(texts)
        all_digits = re.sub(r"\D", "", all_txt)

        print(f"=== h={h} ({mt}) gt#{gi} pred#{pi}  crop={cp.name} ===")
        print(f"  paddle: {[t for t,s in zip(texts,scores) if s > 0.5][:12]}")

        for f in ("id_sku", "print_datetime", "code"):
            v = str(g.get(f, "")).strip()
            if v and v != "nan" and v != "нет":
                seen_field[f] += 1
                digits = re.sub(r"\D", "", v)
                if len(digits) >= 6:
                    if digits in all_digits:
                        by_field[f] += 1
                        print(f"  ✓ {f:<15} GT={digits} FOUND (full)")
                        hits_total += 1
                    elif digits[:8] in all_digits:
                        by_field[f] += 1
                        print(f"  ≈ {f:<15} GT={digits} PARTIAL prefix-8")
                        hits_total += 1
                    elif digits[:6] in all_digits:
                        print(f"  · {f:<15} GT={digits} weak prefix-6")
                    else:
                        print(f"  ✗ {f:<15} GT={digits}")

        for f in ("product_name", "additional_info"):
            v = str(g.get(f, "")).strip()
            if v and v not in ("", "nan", "нет"):
                seen_field[f] += 1
                gt_toks = set(re.findall(r"[A-Za-zА-Яа-я]{4,}", v.lower()))
                pd_toks = set(re.findall(r"[A-Za-zА-Яа-я]{4,}", all_txt.lower()))
                common = gt_toks & pd_toks
                if not common and gt_toks and pd_toks:
                    for gt_t in gt_toks:
                        for pd_t in pd_toks:
                            if gt_t[:4] == pd_t[:4]:
                                common.add(gt_t)
                                break
                if common:
                    by_field[f] += 1
                    print(f"  ✓ {f:<15} GT='{v[:30]}' common toks: {list(common)[:3]}")
                    hits_total += 1
                else:
                    print(f"  ✗ {f:<15} GT='{v[:30]}'")

        v = str(g.get("discount_amount", "")).strip()
        if v and v not in ("", "nan", "нет"):
            seen_field["discount_amount"] += 1
            m = re.search(r"(-?\d{1,3})\s*%", all_txt)
            if m:
                gt_pct = re.search(r"(\d+)", v).group(1)
                if gt_pct in m.group(0):
                    by_field["discount_amount"] += 1
                    print(f"  ✓ discount       GT={v} got {m.group(0)}")
                    hits_total += 1
                else:
                    print(f"  ≈ discount       GT={v} but got {m.group(0)}")
            else:
                print(f"  ✗ discount       GT={v}")
        print()

    print("=" * 60)
    print(f"Per-field hit rate (Paddle vs EasyOCR=0%):")
    for f, n in by_field.items():
        s = seen_field[f]
        if s:
            print(f"  {f:<18s} {n}/{s} = {n/s:.0%}")
    print(f"\nTotal new hits across 8 near-miss pairs: {hits_total}")
    if hits_total >= 6:
        print("→ МАСШТАБИРУЕМ Paddle на все crops (potentially +3-5 пар через порог)")
    elif hits_total >= 3:
        print("→ Стоит попробовать масштабировать")
    else:
        print("→ Слабый эффект, не масштабируем")


if __name__ == "__main__":
    main()
