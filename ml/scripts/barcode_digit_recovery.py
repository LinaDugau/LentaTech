"""Восстановление barcode из OCR-цифр по всем K кадрам трека."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from build_final_csv import choose_best_barcode, fix_ocr_digits, _ean_checksum_ok  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")
TEXT_COLS = ("ocr_text", "paddle_text", "bottom_extra_text", "barcode_raw")
TEXT_COLS_BOTTOM_ONLY = ("bottom_extra_text", "barcode_raw")


def _load_catalog_barcodes(catalog_path: Path | None) -> set[str]:
    if catalog_path is None or not catalog_path.is_file():
        return set()
    out: set[str] = set()
    if catalog_path.suffix.lower() == ".parquet":
        cat = pd.read_parquet(catalog_path)
        col = next((c for c in ("code", "barcode", "ean") if c in cat.columns), None)
        if col:
            for v in cat[col].dropna():
                s = re.sub(r"\D", "", str(v))
                if len(s) >= 12:
                    out.add(s[:13])
                    out.add(s[:12])
        return out
    import csv
    enc = "cp1251" if catalog_path.suffix.lower() == ".csv" else "utf-8"
    sep = ";" if catalog_path.suffix.lower() == ".csv" else ","
    with catalog_path.open(encoding=enc, errors="replace") as f:
        r = csv.reader(f, delimiter=sep)
        hdr = next(r, None)
        if not hdr:
            return out
        ci = next(
            (i for i, c in enumerate(hdr) if str(c).strip().lower() in ("code", "barcode", "ean")),
            1 if len(hdr) > 1 else 0,
        )
        for row in r:
            if ci >= len(row):
                continue
            s = re.sub(r"\D", "", str(row[ci]))
            if len(s) >= 12:
                out.add(s[:13])
                out.add(s[:12])
    return out


def _digit_candidates(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    clean = fix_ocr_digits(text)
    found: list[str] = []
    for m in re.finditer(r"(?<!\d)(\d{12,13})(?!\d)", clean):
        found.append(m.group(1))
    digits = re.sub(r"\D", "", clean)
    for n in (13, 12):
        if len(digits) >= n:
            for i in range(0, len(digits) - n + 1):
                found.append(digits[i: i + n])
    return found


def _score_candidate(code: str, catalog: set[str], count: int) -> tuple:
    if len(code) < 12:
        return (-1, 0, 0, "")
    c13 = code if len(code) == 13 else code[:13]
    chk = _ean_checksum_ok(c13) if len(c13) == 13 else False
    in_cat = c13 in catalog or code[:12] in catalog
    if not chk:
        return (-1, 0, 0, "")
    if not in_cat:
        return (-1, 0, 0, "")
    return (count, len(code), 1 if in_cat else 0, code)


def best_barcode_from_texts(texts: list[str], catalog: set[str], min_votes: int = 2) -> str:
    cands: list[str] = []
    for t in texts:
        cands.extend(_digit_candidates(t))
    if not cands:
        return ""
    counts = Counter(cands)
    ranked = sorted(
        [c for c in counts if counts[c] >= min_votes],
        key=lambda c: (_score_candidate(c, catalog, counts[c])),
        reverse=True,
    )
    if ranked:
        return ranked[0]
    bottomish = [t for t in texts if t]
    if len(bottomish) == 1:
        for c in sorted(counts.keys(), key=lambda c: _score_candidate(c, catalog, counts[c]), reverse=True):
            if _score_candidate(c, catalog, counts[c])[0] >= 0:
                return c
    return ""


def recover_video(
    video: str,
    catalog: set[str],
    in_place: bool = True,
    ocr_path: Path | None = None,
    text_cols: tuple[str, ...] = TEXT_COLS,
    min_votes: int = 2,
) -> int:
    path = ocr_path or (ROOT / "ml" / "output" / f"ocr_qr_{video}.csv")
    if not path.is_file():
        print(f"  skip {video}: no ocr")
        return 0
    df = pd.read_csv(path)
    if "track_id" not in df.columns:
        return 0
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    updated = 0
    for tid, sub in df.groupby("track_id"):
        existing = choose_best_barcode(
            [str(x) for x in sub["barcode_raw"].tolist() if str(x) not in ("", "nan")]
        )
        if len(re.sub(r"\D", "", existing)) >= 12 and _ean_checksum_ok(
            re.sub(r"\D", "", existing)[:13]
        ):
            continue
        texts: list[str] = []
        for _, row in sub.iterrows():
            for col in text_cols:
                if col not in sub.columns:
                    continue
                v = row.get(col, "")
                if isinstance(v, str) and v.strip():
                    texts.append(v)
        bc = best_barcode_from_texts(texts, catalog, min_votes=min_votes)
        if len(re.sub(r"\D", "", bc)) < 12:
            continue
        mask = df["track_id"] == tid
        df.loc[mask, "barcode_raw"] = bc
        updated += 1

    out = path if in_place else path.with_name(f"ocr_qr_{video}_bcdig.csv")
    df.to_csv(out, index=False)
    print(f"  {video}: {updated} tracks recovered → {out.name}")
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--ocr", type=Path, default=None, help="путь к ocr_qr CSV")
    ap.add_argument("--bottom-only", action="store_true", help="только bottom_extra_text")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--no-in-place", action="store_true")
    args = ap.parse_args()
    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["26_12-20"])
    catalog = _load_catalog_barcodes(args.catalog)
    print(f"Catalog barcodes: {len(catalog)}")
    cols = TEXT_COLS_BOTTOM_ONLY if args.bottom_only else TEXT_COLS
    total = 0
    for v in videos:
        total += recover_video(
            v, catalog, in_place=not args.no_in_place, ocr_path=args.ocr,
            text_cols=cols, min_votes=args.min_votes,
        )
    print(f"Total recovered: {total}")


if __name__ == "__main__":
    main()
