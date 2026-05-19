"""Temporal barcode decode: full video crop на каждом K-кадре трека."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_recovery_probe import decode_image  # noqa: E402
from build_final_csv import choose_best_barcode, _ean_checksum_ok  # noqa: E402
from video_resolve import find_video  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")


def _load_catalog(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
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


def _expand_box(x1, y1, x2, y2, w, h, pad_x=0.30, pad_down=0.75):
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    return (
        max(0, int(x1 - bw * pad_x)),
        max(0, int(y1 - bh * 0.12)),
        min(w - 1, int(x2 + bw * pad_x)),
        min(h - 1, int(y2 + bh * pad_down)),
    )


def _video_crop(cap, ts_ms: float, box, w: int, h: int):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    x1, y1, x2, y2 = _expand_box(*box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _score_bc(code: str, catalog: set[str], votes: int) -> tuple:
    s = re.sub(r"\D", "", code)
    if len(s) < 12:
        return (-1,)
    c13 = s[:13] if len(s) >= 13 else s
    chk = len(c13) == 13 and _ean_checksum_ok(c13)
    in_cat = c13 in catalog or s[:12] in catalog
    return (votes, int(chk), int(in_cat), len(s))


def decode_track(
    cap,
    w: int,
    h: int,
    rows: pd.DataFrame,
    catalog: set[str],
    fast: bool = False,
) -> str:
    hits: Counter[str] = Counter()
    for _, r in rows.iterrows():
        ts = float(r.frame_ts_ms)
        box = (
            float(r.x_min_orig), float(r.y_min_orig),
            float(r.x_max_orig), float(r.y_max_orig),
        )
        crop = _video_crop(cap, ts, box, w, h)
        if crop is None:
            continue
        for code in decode_image(crop, fast=fast):
            if re.fullmatch(r"\d{12,13}", code):
                hits[code] += 1
        ch, cw = crop.shape[:2]
        bot = crop[int(ch * 0.45):, :]
        if bot.size:
            for code in decode_image(bot, fast=True):
                if re.fullmatch(r"\d{12,13}", code):
                    hits[code] += 1
    if not hits:
        return ""
    ranked = sorted(hits.keys(), key=lambda c: _score_bc(c, catalog, hits[c]), reverse=True)
    best = ranked[0]
    c13 = best[:13]
    if len(c13) == 13 and _ean_checksum_ok(c13) and (c13 in catalog or best[:12] in catalog):
        return c13
    return ""


def process_video(video: str, catalog: set[str], fast: bool = False, ocr_path: Path | None = None) -> int:
    path = ocr_path or (ROOT / "ml" / "output" / f"ocr_qr_{video}.csv")
    if not path.is_file():
        print(f"  skip {video}: no ocr")
        return 0
    vpath = find_video(f"{video}.mp4")
    if vpath is None:
        print(f"  skip {video}: no video")
        return 0

    df = pd.read_csv(path)
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    cap = cv2.VideoCapture(str(vpath))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    updated = 0
    groups = list(df.groupby("track_id"))
    for tid, sub in tqdm(groups, desc=video):
        existing = choose_best_barcode([str(x) for x in sub["barcode_raw"] if str(x) not in ("", "nan")])
        if len(re.sub(r"\D", "", existing)) >= 12:
            continue
        bc = decode_track(cap, w, h, sub, catalog, fast=fast)
        if len(re.sub(r"\D", "", bc)) < 12:
            continue
        df.loc[df["track_id"] == tid, "barcode_raw"] = bc
        updated += 1

    cap.release()
    df.to_csv(path, index=False)
    print(f"  {video}: {updated} tracks with video barcode → {path.name}")
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true", help="меньше поворотов decode")
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--ocr", type=Path, default=None)
    args = ap.parse_args()
    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["25_12-20"])
    catalog = _load_catalog(args.catalog)
    print(f"Catalog: {len(catalog)} codes")
    total = 0
    for v in videos:
        total += process_video(v, catalog, fast=args.fast, ocr_path=args.ocr)
    print(f"Total tracks updated: {total}")


if __name__ == "__main__":
    main()
