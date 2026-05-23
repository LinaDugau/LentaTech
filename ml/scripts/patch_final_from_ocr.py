"""Патч final_eval из OCR всех K кадров."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ("25_12-20", "26_12-20", "43_15")

PATCH_FIELDS = (
    "id_sku", "print_datetime", "code", "special_symbols",
    "additional_info", "discount_amount",
)


def _empty(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no", "n/a")


def _track_fused(final_row, ocr_df):
    from multi_frame_field_fusion import fuse_track

    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(str(final_row.get("x_min", 0)).replace(",", "."))
    y0 = float(str(final_row.get("y_min", 0)).replace(",", "."))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
    ]
    if cand.empty:
        return {}
    tid = int(cand.iloc[0]["track_id"])
    sub = ocr_df[ocr_df["track_id"] == tid]
    if sub.empty:
        return {}
    return fuse_track(sub)


def patch_file(final_path: Path, ocr_path: Path) -> int:
    if not final_path.is_file() or not ocr_path.is_file():
        return 0
    df = pd.read_csv(final_path)
    ocr = pd.read_csv(ocr_path)
    changed = 0
    for idx, row in df.iterrows():
        fused = _track_fused(row, ocr)
        if not fused:
            continue
        for field in PATCH_FIELDS:
            if field not in df.columns or not _empty(row.get(field)):
                continue
            val = str(fused.get(field, "") or "").strip()
            if val and not _empty(val):
                df.at[idx, field] = val
                changed += 1
    if changed:
        df.to_csv(final_path, index=False)
    print(f"  {final_path.name}: {changed} field fills")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="append")
    ap.add_argument("--dir", type=Path, default=None, help="директория с final/ocr CSV")
    args = ap.parse_args()
    base = args.dir or (ROOT / "ml" / "output")
    videos = args.video or (list(VIDEOS) if args.all else ["26_12-20"])
    total = 0
    for v in videos:
        p = base / f"final_eval_{v}.csv"
        ocr = base / f"ocr_qr_{v}.csv"
        total += patch_file(p, ocr)
    print(f"Total fills: {total}")


if __name__ == "__main__":
    main()
