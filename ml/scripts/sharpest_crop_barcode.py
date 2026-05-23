"""Barcode decode на 4K crop (sharpest-frame per GT-track)."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_recovery_probe import decode_image, _expand_for_barcode  # noqa: E402
from build_final_csv import choose_best_barcode, _ean_checksum_ok  # noqa: E402
from extra_top_ocr import (  # noqa: E402
    _VideoReader,
    _crop_from_row,
    _pick_track_rows,
    _track_ids_from_gt,
    find_video,
)
from whole_frame_codes2 import _propagate_tracks  # noqa: E402


def _load_catalog(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    out: set[str] = set()
    if path.suffix.lower() == ".parquet":
        cat = pd.read_parquet(path)
        col = next((c for c in ("code", "barcode", "ean") if c in cat.columns), None)
        if col:
            for v in cat[col].dropna():
                s = re.sub(r"\D", "", str(v))
                if len(s) >= 12:
                    out.add(s[:13])
                    out.add(s[:12])
        return out
    import csv
    with path.open(encoding="cp1251", errors="replace") as f:
        r = csv.reader(f, delimiter=";")
        hdr = next(r, None)
        if not hdr:
            return out
        ci = next((i for i, c in enumerate(hdr) if str(c).strip().lower() == "code"), 1)
        for row in r:
            if ci < len(row):
                s = re.sub(r"\D", "", row[ci])
                if len(s) >= 12:
                    out.add(s[:13])
                    out.add(s[:12])
    return out


def _expand_crop(reader: _VideoReader, row):
    """Crop с расширением вниз под штрихкод."""
    frame = reader.frame_at_ms(float(row["frame_ts_ms"]))
    if frame is None:
        return None
    x1, y1, x2, y2 = (
        float(row["x_min_orig"]), float(row["y_min_orig"]),
        float(row["x_max_orig"]), float(row["y_max_orig"]),
    )
    ex1, ey1, ex2, ey2 = _expand_for_barcode(x1, y1, x2, y2, reader.w, reader.h)
    crop = frame[ey1:ey2, ex1:ex2]
    return crop if crop.size else None


def _decode_crop(crop, fast: bool = False) -> Counter[str]:
    hits: Counter[str] = Counter()
    if crop is None or crop.size == 0:
        return hits
    for code in decode_image(crop, fast=fast):
        if re.fullmatch(r"\d{12,13}", code):
            hits[code] += 1
    h, w = crop.shape[:2]
    for y0 in (0.40, 0.50, 0.60):
        bot = crop[int(h * y0):, :]
        if bot.size:
            for code in decode_image(bot, fast=True):
                if re.fullmatch(r"\d{12,13}", code):
                    hits[code] += 1
    return hits


def _pick_barcode(hits: Counter[str], catalog: set[str], require_catalog: bool) -> str:
    if not hits:
        return ""
    ranked = sorted(
        hits.keys(),
        key=lambda c: (
            hits[c],
            int(_ean_checksum_ok(c[:13])),
            int(c in catalog or c[:12] in catalog),
        ),
        reverse=True,
    )
    for code in ranked:
        c13 = code[:13]
        if len(c13) == 13 and _ean_checksum_ok(c13):
            if not require_catalog or c13 in catalog or code[:12] in catalog:
                return c13
    return ""


def process(
    csv_path: Path,
    gt_path: Path | None,
    catalog: set[str],
    *,
    force: bool = False,
    sharpest: bool = True,
    all_frames: bool = False,
    fast: bool = False,
    require_catalog: bool = True,
    track_ids: set[int] | None = None,
) -> tuple[int, int, int]:
    full_df = pd.read_csv(csv_path)
    if full_df.empty:
        return 0, 0, 0
    if "barcode_raw" not in full_df.columns:
        full_df["barcode_raw"] = ""
    full_df["barcode_raw"] = full_df["barcode_raw"].astype(object)

    if track_ids is None and gt_path:
        track_ids = _track_ids_from_gt(gt_path, csv_path)
        print(f"  --gt: {len(track_ids)} track_ids")

    df = full_df
    if track_ids:
        df = full_df[full_df["track_id"].astype(int).isin(track_ids)].copy()

    video_name = str(df.iloc[0].get("video", "") or "").strip()
    vpath = find_video(video_name)
    if vpath is None:
        print(f"  video not found: {video_name}")
        return 0, 0, 0
    reader = _VideoReader(vpath)
    print(f"  source: {vpath} ({reader.w}x{reader.h})")

    gt_bc_map: dict[int, str] = {}
    if gt_path and gt_path.is_file():
        gt = pd.read_csv(gt_path)
        ocr = full_df
        from extra_top_ocr import _to_float, _iou
        for gi, g in gt.iterrows():
            gb = (_to_float(g["x_min"]), _to_float(g["y_min"]),
                  _to_float(g["x_max"]), _to_float(g["y_max"]))
            best_iou, best_tid = 0.0, None
            for tid in (track_ids or set()):
                sub = ocr[ocr["track_id"] == tid]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                ob = (float(r.x_min_orig), float(r.y_min_orig),
                      float(r.x_max_orig), float(r.y_max_orig))
                v = _iou(gb, ob)
                if v > best_iou:
                    best_iou, best_tid = v, tid
            if best_tid is not None and best_iou >= 0.25:
                gt_bc_map[best_tid] = re.sub(r"\D", "", str(g.get("barcode", "")))

    n_updated = 0
    n_gt_hit = 0
    n_any_decode = 0

    try:
        groups = list(df.groupby("track_id"))
        for tid, sub in tqdm(groups, desc=csv_path.stem):
            existing = choose_best_barcode(
                [str(x) for x in sub["barcode_raw"] if str(x).strip() not in ("", "nan")]
            )
            if not force and len(re.sub(r"\D", "", existing)) >= 12:
                continue

            hits: Counter[str] = Counter()
            if all_frames:
                rows = sub
            elif sharpest:
                rows = _pick_track_rows(sub, sharpest=True, reader=reader)
            else:
                rows = _pick_track_rows(sub, sharpest=False, reader=reader)

            for _, row in rows.iterrows():
                crop = _expand_crop(reader, row)
                if crop is None:
                    crop = _crop_from_row(row, reader)
                hits += _decode_crop(crop, fast=fast)

            bc = _pick_barcode(hits, catalog, require_catalog)
            if hits:
                n_any_decode += 1
            if bc:
                full_df.loc[full_df["track_id"] == tid, "barcode_raw"] = bc
                n_updated += 1
                gtbc = gt_bc_map.get(int(tid), "")
                if gtbc and (bc == gtbc or bc.startswith(gtbc[:12]) or gtbc.startswith(bc[:12])):
                    n_gt_hit += 1
    finally:
        reader.close()

    _propagate_tracks(full_df)
    full_df.to_csv(csv_path, index=False)
    print(
        f"  {csv_path.name}: decode_any={n_any_decode}, updated={n_updated}, "
        f"gt_exact={n_gt_hit}/{len(gt_bc_map)}"
    )
    return n_any_decode, n_updated, n_gt_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--gt", type=str, default=None)
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-frames", action="store_true", help="все K кадров трека")
    ap.add_argument("--no-sharpest", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-require-catalog", action="store_true")
    args = ap.parse_args()
    catalog = _load_catalog(args.catalog)
    print(f"  catalog: {len(catalog)} codes, zxing via barcode_recovery_probe")
    files = [Path(p) for p in args.files] if args.files else [
        ROOT / "ml" / "output" / f"ocr_qr_{v}.csv" for v in ("25_12-20",)
    ]
    gt = Path(args.gt) if args.gt else None
    total = 0
    for f in files:
        print(f"\n=== {f.name} ===")
        _, upd, _ = process(
            f, gt, catalog,
            force=args.force,
            sharpest=not args.no_sharpest,
            all_frames=args.all_frames,
            fast=args.fast,
            require_catalog=not args.no_require_catalog,
        )
        total += upd
    print(f"\nTotal tracks updated: {total}")


if __name__ == "__main__":
    main()
