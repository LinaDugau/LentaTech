"""Partial EAN из OCR-цифр."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from build_final_csv import choose_best_barcode, fix_ocr_digits, _ean_checksum_ok  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")
TEXT_COLS = ("ocr_text", "paddle_text", "bottom_extra_text", "barcode_raw", "top_extra_text")


def ean13_check_digit(d12: str) -> str:
    if len(d12) != 12 or not d12.isdigit():
        return ""
    digits = [int(c) for c in d12]
    chk = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return str(chk)


def _load_catalog(path: Path) -> tuple[set[str], dict[str, list[str]]]:
    full: set[str] = set()
    prefix: dict[str, list[str]] = defaultdict(list)
    if not path.is_file():
        return full, prefix
    import csv
    enc = "cp1251" if path.suffix.lower() == ".csv" else "utf-8"
    sep = ";" if path.suffix.lower() == ".csv" else ","
    with path.open(encoding=enc, errors="replace") as f:
        r = csv.reader(f, delimiter=sep)
        hdr = next(r, None)
        if not hdr:
            return full, prefix
        ci = next(
            (i for i, c in enumerate(hdr) if str(c).strip().lower() in ("code", "barcode", "ean")),
            1 if len(hdr) > 1 else 0,
        )
        for row in r:
            if ci >= len(row):
                continue
            s = re.sub(r"\D", "", str(row[ci]))
            if len(s) == 13 and _ean_checksum_ok(s):
                full.add(s)
                for n in (10, 11, 12):
                    prefix[s[:n]].append(s)
            elif len(s) == 12:
                full.add(s)
                for n in (10, 11):
                    prefix[s[:n]].append(s)
    return full, dict(prefix)


def _digit_runs(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    clean = fix_ocr_digits(text)
    runs: list[str] = []
    for m in re.finditer(r"(?<!\d)(\d{8,13})(?!\d)", clean):
        runs.append(m.group(1))
    digits = re.sub(r"\D", "", clean)
    for n in (13, 12, 11, 10):
        if len(digits) >= n:
            for i in range(0, len(digits) - n + 1):
                runs.append(digits[i: i + n])
    return runs


def _complete_candidates(run: str, catalog: set[str], prefix_idx: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    if len(run) == 13 and _ean_checksum_ok(run) and run in catalog:
        return [run]
    if len(run) == 12:
        full = run + ean13_check_digit(run)
        if len(full) == 13 and _ean_checksum_ok(full) and full in catalog:
            out.append(full)
        for ean in prefix_idx.get(run, []):
            if ean in catalog:
                out.append(ean)
    elif 10 <= len(run) <= 11:
        for ean in prefix_idx.get(run, []):
            if ean in catalog:
                out.append(ean)
    return out


def _hamming1_fix(run12: str, catalog: set[str]) -> list[str]:
    """Одна OCR-ошибка в 12 цифрах → валидный EAN в каталоге."""
    if len(run12) != 12:
        return []
    found: list[str] = []
    for pos in range(12):
        for d in "0123456789":
            if d == run12[pos]:
                continue
            trial = run12[:pos] + d + run12[pos + 1:]
            full = trial + ean13_check_digit(trial)
            if len(full) == 13 and _ean_checksum_ok(full) and full in catalog:
                found.append(full)
    return found


def best_partial_barcode(
    texts: list[str],
    catalog: set[str],
    prefix_idx: dict[str, list[str]],
    *,
    min_votes: int = 1,
    allow_hamming1: bool = True,
) -> str:
    votes: Counter[str] = Counter()
    for text in texts:
        for run in _digit_runs(text):
            for ean in _complete_candidates(run, catalog, prefix_idx):
                votes[ean] += 1
            if allow_hamming1 and len(run) == 12:
                for ean in _hamming1_fix(run, catalog):
                    votes[ean] += 1
    if not votes:
        return ""
    ranked = sorted(votes.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    best, count = ranked[0]
    if count < min_votes:
        return ""
    return best


def recover_file(
    ocr_path: Path,
    catalog_path: Path,
    *,
    force: bool = False,
    min_votes: int = 1,
    allow_hamming1: bool = True,
) -> int:
    if not ocr_path.is_file():
        print(f"  skip missing {ocr_path}")
        return 0
    catalog, prefix_idx = _load_catalog(catalog_path)
    print(f"  catalog={len(catalog)} EAN, prefix keys={len(prefix_idx)}")

    df = pd.read_csv(ocr_path)
    if "track_id" not in df.columns:
        return 0
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    updated = 0
    for tid, sub in df.groupby("track_id"):
        existing = choose_best_barcode(
            [str(x) for x in sub["barcode_raw"] if str(x).strip() not in ("", "nan")]
        )
        ex = re.sub(r"\D", "", existing)
        if not force and len(ex) >= 12 and (len(ex) == 13 and _ean_checksum_ok(ex[:13])):
            continue
        texts: list[str] = []
        for _, row in sub.iterrows():
            for col in TEXT_COLS:
                if col not in sub.columns:
                    continue
                v = row.get(col, "")
                if isinstance(v, str) and v.strip():
                    texts.append(v)
        bc = best_partial_barcode(
            texts, catalog, prefix_idx,
            min_votes=min_votes,
            allow_hamming1=allow_hamming1,
        )
        if len(re.sub(r"\D", "", bc)) < 12:
            continue
        df.loc[df["track_id"] == tid, "barcode_raw"] = bc
        updated += 1

    df.to_csv(ocr_path, index=False)
    print(f"  {ocr_path.name}: partial recovered={updated}")
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ocr", type=Path, default=None, help="ocr_qr CSV")
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-votes", type=int, default=1)
    ap.add_argument("--no-hamming1", action="store_true")
    args = ap.parse_args()

    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["26_12-20"])
    total = 0
    for v in videos:
        path = args.ocr or (ROOT / "ml" / "output" / f"ocr_qr_{v}.csv")
        print(f"\n=== {v} ===")
        total += recover_file(
            path, args.catalog,
            force=args.force,
            min_votes=args.min_votes,
            allow_hamming1=not args.no_hamming1,
        )
    print(f"\nTotal partial recovered: {total}")


if __name__ == "__main__":
    main()
