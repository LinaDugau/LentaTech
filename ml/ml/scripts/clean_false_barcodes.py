"""Очистка ложных barcode_raw после агрессивного digit recovery."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_digit_recovery import _load_catalog_barcodes  # noqa: E402
from build_final_csv import _ean_checksum_ok  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")


def _ok(bc: str, catalog: set[str]) -> bool:
    s = re.sub(r"\D", "", str(bc or ""))
    if len(s) < 12:
        return False
    c13 = s[:13] if len(s) >= 13 else s
    if len(c13) == 13 and not _ean_checksum_ok(c13):
        return False
    return c13 in catalog or s[:12] in catalog


def clean(path: Path, catalog: set[str]) -> int:
    df = pd.read_csv(path)
    if "barcode_raw" not in df.columns:
        return 0
    df["barcode_raw"] = df["barcode_raw"].astype(object)
    cleared = 0
    for idx, val in df["barcode_raw"].items():
        s = str(val or "").strip()
        if not s or s == "nan":
            continue
        if not _ok(s, catalog):
            df.at[idx, "barcode_raw"] = ""
            cleared += 1
    df.to_csv(path, index=False)
    return cleared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="append")
    args = ap.parse_args()
    catalog = _load_catalog_barcodes(ROOT / "db_hack.csv")
    videos = args.video or (list(VIDEOS) if args.all else list(VIDEOS))
    for v in videos:
        p = ROOT / "ml" / "output" / f"ocr_qr_{v}.csv"
        n = clean(p, catalog)
        print(f"  {v}: cleared {n} false barcode_raw")


if __name__ == "__main__":
    main()
