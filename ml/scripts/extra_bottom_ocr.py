"""OCR нижней зоны ценника (barcode / id_sku / print_datetime)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from build_final_csv import fix_ocr_digits, _ean_checksum_ok  # noqa: E402
from extra_top_ocr import (  # noqa: E402
    _VideoReader,
    _crop_from_row,
    _pick_track_rows,
    _track_ids_from_final,
    _track_ids_from_gt,
    find_video,
    get_reader,
)

_READER = None
_DIGIT_RX = re.compile(r"(?<!\d)(\d{12,13})(?!\d)")


def preprocess_bottom(img: np.ndarray, bottom_pct: float = 0.35, upscale: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    y_start = max(0, int(h * (1 - bottom_pct)))
    bottom = img[y_start:, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    if upscale > 1:
        eq = cv2.resize(
            eq,
            (eq.shape[1] * upscale, eq.shape[0] * upscale),
            interpolation=cv2.INTER_CUBIC,
        )
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def ocr_bottom_image(img: np.ndarray, allowlist: str = "0123456789-:./ ") -> str:
    if img is None or img.size == 0:
        return ""
    proc = preprocess_bottom(img)
    if proc.shape[0] < 16 or proc.shape[1] < 16:
        return ""
    try:
        results = get_reader().readtext(proc, detail=1, paragraph=False, allowlist=allowlist)
    except Exception:
        return ""
    tokens = [str(t).strip() for (_, t, _) in results if t and str(t).strip()]
    return " ".join(tokens)


def _extract_ean(text: str) -> str:
    clean = fix_ocr_digits(str(text or ""))
    for m in _DIGIT_RX.finditer(clean):
        code = m.group(1)
        c13 = code if len(code) == 13 else code[:13]
        if len(c13) == 13 and _ean_checksum_ok(c13):
            return c13
    digits = re.sub(r"\D", "", clean)
    for n in (13, 12):
        if len(digits) >= n:
            for i in range(len(digits) - n + 1):
                chunk = digits[i: i + n]
                c13 = chunk if len(chunk) == 13 else chunk[:13]
                if len(c13) == 13 and _ean_checksum_ok(c13):
                    return c13
    return ""


def _load_catalog_barcodes(catalog_path: Path | None) -> set[str]:
    if catalog_path is None or not catalog_path.is_file():
        return set()
    out: set[str] = set()
    if catalog_path.suffix.lower() == ".parquet":
        import pandas as pd
        cat = pd.read_parquet(catalog_path)
        col = "code" if "code" in cat.columns else "barcode" if "barcode" in cat.columns else None
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
        ci = next((i for i, c in enumerate(hdr) if str(c).strip().lower() in ("code", "barcode", "ean")), 1)
        for row in r:
            if ci >= len(row):
                continue
            s = re.sub(r"\D", "", str(row[ci]))
            if len(s) >= 12:
                out.add(s[:13])
                out.add(s[:12])
    return out


def process(
    csv_path: Path,
    force: bool = False,
    per_track: bool = False,
    source: str = "auto",
    limit: int | None = None,
    track_ids: set[int] | None = None,
    sharpest: bool = False,
    catalog: set[str] | None = None,
    write_barcode: bool = True,
) -> None:
    full_df = pd.read_csv(csv_path)
    if full_df.empty:
        return
    if "bottom_extra_text" not in full_df.columns:
        full_df["bottom_extra_text"] = ""
    if "barcode_raw" not in full_df.columns:
        full_df["barcode_raw"] = ""

    df = full_df
    if track_ids:
        df = full_df[full_df["track_id"].astype(int).isin(track_ids)].copy()
        if df.empty:
            print("  no rows for requested track_ids")
            return
        print(f"  filtered to {df.track_id.nunique()} tracks ({len(df)} rows)")

    video_name = str(df.iloc[0].get("video", "") or "").strip()
    reader: _VideoReader | None = None
    if source in ("auto", "video") and video_name:
        vpath = find_video(video_name)
        if vpath is not None:
            reader = _VideoReader(vpath)
            print(f"  source video: {vpath} ({reader.w}x{reader.h})")

    work = _pick_track_rows(df, sharpest=sharpest, reader=reader) if per_track else df
    if sharpest and per_track:
        print(f"  sharpest-frame per track: {len(work)} rows")
    if limit:
        work = work.head(limit)

    track_text: dict[int, str] = {}
    track_bc: dict[int, str] = {}
    n_new = 0
    n_bc = 0

    try:
        for _, row in tqdm(work.iterrows(), total=len(work), desc=csv_path.stem):
            idx = row.name
            if str(full_df.at[idx, "bottom_extra_text"] or "").strip() and not force:
                continue
            use_reader = reader if source != "hires" else None
            img = _crop_from_row(row, use_reader)
            text = ocr_bottom_image(img)
            tid = int(row["track_id"]) if per_track and "track_id" in row.index else None
            if tid is not None:
                track_text[tid] = text
                if write_barcode and catalog:
                    bc = _extract_ean(text)
                    if bc and (bc in catalog or bc[:12] in catalog):
                        track_bc[tid] = bc
            else:
                full_df.at[idx, "bottom_extra_text"] = text
            n_new += 1

        if per_track and track_text:
            for idx, row in full_df.iterrows():
                tid = int(row["track_id"])
                if tid not in track_text:
                    continue
                if str(full_df.at[idx, "bottom_extra_text"] or "").strip() and not force:
                    continue
                full_df.at[idx, "bottom_extra_text"] = track_text[tid]
                if tid in track_bc and not str(full_df.at[idx, "barcode_raw"] or "").strip():
                    full_df.at[idx, "barcode_raw"] = track_bc[tid]
                    n_bc += 1
    finally:
        if reader is not None:
            reader.close()

    full_df.to_csv(csv_path, index=False)
    total_nonempty = sum(1 for v in full_df["bottom_extra_text"] if str(v).strip())
    print(
        f"  {csv_path.name}: new={n_new}, bottom_nonempty={total_nonempty}/{len(full_df)}, "
        f"barcode_raw+={n_bc}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--per-track", action="store_true")
    ap.add_argument("--source", choices=("auto", "video", "hires"), default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--final", type=str, default=None)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--sharpest", action="store_true")
    ap.add_argument("--catalog", type=Path, default=ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet")
    ap.add_argument("--no-write-barcode", action="store_true")
    args = ap.parse_args()
    catalog = _load_catalog_barcodes(args.catalog)
    print(f"  catalog barcodes: {len(catalog)}")
    files = [Path(p) for p in args.files] if args.files else sorted(
        (ROOT / "ml" / "output").glob("ocr_qr_*.csv")
    )
    files = [f for f in files if "_2-10" not in f.name and f.stem != "ocr_qr_input"]
    for f in files:
        print(f"\n=== {f.name} ===")
        tids = None
        if args.gt:
            tids = _track_ids_from_gt(Path(args.gt), f)
            print(f"  --gt: {len(tids)} track_ids")
        elif args.final:
            tids = _track_ids_from_final(Path(args.final), f)
            print(f"  --final: {len(tids)} track_ids")
        process(
            f,
            force=args.force,
            per_track=args.per_track,
            source=args.source,
            limit=args.limit,
            track_ids=tids,
            sharpest=args.sharpest,
            catalog=catalog,
            write_barcode=not args.no_write_barcode,
        )


if __name__ == "__main__":
    main()
