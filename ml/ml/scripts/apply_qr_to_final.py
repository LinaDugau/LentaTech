"""Прописать barcode/QR-поля из ocr_qr в final_eval (без полного rebuild)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "build_final_csv", ROOT / "ml" / "scripts" / "build_final_csv.py"
)
_bfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bfc)

VIDEOS = ("25_12-20", "26_12-20", "43_15")

QR_OUT_FIELDS = (
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
)


def _empty(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no", "n/a")


def _track_qr(ocr_df: pd.DataFrame, final_row) -> tuple[str, str]:
    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(str(final_row.get("x_min", 0)).replace(",", "."))
    y0 = float(str(final_row.get("y_min", 0)).replace(",", "."))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
    ]
    if cand.empty:
        return "", ""
    tid = int(cand.iloc[0]["track_id"])
    sub = ocr_df[ocr_df["track_id"] == tid]
    qrs = [str(x) for x in sub.get("qr_raw", []) if str(x).strip() and str(x) != "nan"]
    bcs = [str(x) for x in sub.get("barcode_raw", []) if str(x).strip() and str(x) != "nan"]
    return _bfc.choose_best_qr(qrs), _bfc.choose_best_barcode(bcs)


def apply_file(final_path: Path, ocr_path: Path) -> int:
    if not final_path.is_file() or not ocr_path.is_file():
        return 0
    df = pd.read_csv(final_path, dtype={"barcode": str}, keep_default_na=False)
    ocr = pd.read_csv(ocr_path)
    changed = 0
    for idx, row in df.iterrows():
        qr_raw, bc_raw = _track_qr(ocr, row)
        if not qr_raw and not bc_raw:
            continue
        qr_fields = _bfc.parse_qr(qr_raw) if qr_raw else {}
        bc = _bfc.parse_barcode("", bc_raw) if bc_raw else ""
        if not bc and qr_fields.get("qr_code_barcode"):
            qbc = str(qr_fields["qr_code_barcode"]).strip()
            if qbc.isdigit() and len(qbc) >= 12:
                bc = qbc
        if bc and _empty(row.get("barcode")):
            df.at[idx, "barcode"] = bc
            changed += 1
        for field in QR_OUT_FIELDS:
            if field not in df.columns:
                continue
            if not _empty(row.get(field)):
                continue
            val = str(qr_fields.get(field, "") or "").strip()
            if val and not _empty(val):
                df.at[idx, field] = val
                changed += 1
    if changed:
        df.to_csv(final_path, index=False)
    print(f"  {final_path.name}: {changed} field updates")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="append")
    ap.add_argument("--final-dir", type=Path, default=None)
    ap.add_argument("--ocr-dir", type=Path, default=None)
    args = ap.parse_args()
    videos = args.video or (list(VIDEOS) if args.all else ["25_12-20"])
    out_dir = args.final_dir or (ROOT / "ml" / "output")
    ocr_dir = args.ocr_dir or (ROOT / "ml" / "output")
    total = 0
    for v in videos:
        total += apply_file(
            out_dir / f"final_eval_{v}.csv",
            ocr_dir / f"ocr_qr_{v}.csv",
        )
    print(f"Total updates: {total}")


if __name__ == "__main__":
    main()
