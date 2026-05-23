"""Инференс digit-strip CNN."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import (  # noqa: E402
    N_DIGITS,
    ean_checksum_ok,
    extract_digit_strip,
    normalize_barcode,
    split_digit_slots,
)
from extra_top_ocr import _VideoReader, _blur_var, _crop_from_row  # noqa: E402
from train_barcode_strip import MODEL_PATH, SlotCNN  # noqa: E402
from barcode_digit_recovery import _load_catalog_barcodes  # noqa: E402
from video_resolve import find_video  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")
Y_BANDS = ((0.48, 0.95), (0.52, 0.92), (0.55, 0.92), (0.40, 0.88), (0.60, 0.96))


def _load_model(device: torch.device) -> SlotCNN:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run train_barcode_strip.py first.")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = SlotCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _predict_digits(model: SlotCNN, strip: np.ndarray, device: torch.device) -> str:
    slots = split_digit_slots(strip, n=N_DIGITS)
    xs = torch.from_numpy(slots.astype(np.float32) / 255.0).unsqueeze(1).to(device)
    with torch.no_grad():
        digits = model(xs).argmax(1).cpu().numpy().tolist()
    return "".join(str(d) for d in digits)


def _best_ean_from_digits(raw: str) -> str:
    candidates: list[str] = []
    if len(raw) >= 13:
        candidates.append(raw[:13])
    if len(raw) >= 12:
        candidates.append(raw[:12])
        candidates.append(raw[1:13])
    for base in list(candidates):
        if len(base) == 12:
            d = [int(c) for c in base]
            check = (10 - ((sum(d[0:11:2]) + 3 * sum(d[1:11:2])) % 10)) % 10
            candidates.append(base + str(check))
    ranked = sorted(
        set(candidates),
        key=lambda c: (ean_checksum_ok(c), len(c), c),
        reverse=True,
    )
    for c in ranked:
        if re.fullmatch(r"\d{12,13}", c):
            if len(c) == 13 and ean_checksum_ok(c):
                return c
            if len(c) == 12:
                return c
    return ranked[0] if ranked else ""


def predict_strip(model: SlotCNN, strip: np.ndarray, device: torch.device) -> str:
    raw = _predict_digits(model, strip, device)
    return _best_ean_from_digits(raw)


def _decode_crop(model, device, crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return ""
    votes: Counter[str] = Counter()
    for y0, y1 in Y_BANDS:
        strip = extract_digit_strip(crop, y_start=y0, y_end=y1)
        if strip is None:
            continue
        bc = predict_strip(model, strip, device)
        norm = normalize_barcode(bc)
        if len(norm) >= 12 and _digit_entropy_ok(norm) and (
            (len(norm) == 13 and ean_checksum_ok(norm))
            or len(norm) == 12
        ):
            votes[norm] += 2 if ean_checksum_ok(norm) else 1
    if not votes:
        return ""
    return votes.most_common(1)[0][0]


def _track_frame_rows(
    df: pd.DataFrame,
    tid: int,
    *,
    sharpest: bool,
    reader: _VideoReader | None,
    max_frames: int,
) -> pd.DataFrame:
    sub = df[df["track_id"] == tid]
    if sub.empty:
        return sub
    if not sharpest or reader is None or len(sub) <= max_frames:
        return sub.head(max_frames) if len(sub) > max_frames else sub
    scored: list[tuple[float, int]] = []
    for idx, row in sub.iterrows():
        img = _crop_from_row(row, reader)
        score = _blur_var(img) if img is not None and img.size else -1.0
        scored.append((score, idx))
    scored.sort(reverse=True)
    picks = [idx for _, idx in scored[:max_frames]]
    return df.loc[picks]


def _digit_entropy_ok(code: str) -> bool:
    """Отсечь «000000…» и прочий мусор CNN."""
    if len(code) < 12:
        return False
    if len(set(code)) <= 3:
        return False
    zeros = code.count("0")
    if zeros >= 6:
        return False
    return True


def _in_catalog(code: str, catalog: set[str]) -> bool:
    if not catalog:
        return True
    norm = normalize_barcode(code)
    if len(norm) < 12:
        return False
    return norm[:13] in catalog or norm[:12] in catalog


def _accept_barcode(code: str, *, min_votes: int, votes: int, catalog: set[str]) -> bool:
    norm = normalize_barcode(code)
    if len(norm) < 12:
        return False
    if not _digit_entropy_ok(norm):
        return False
    if votes < min_votes:
        return False
    if not _in_catalog(norm, catalog):
        return False
    if len(norm) == 13:
        return ean_checksum_ok(norm)
    d = [int(c) for c in norm[:12]]
    check = (10 - ((sum(d[0:11:2]) + 3 * sum(d[1:11:2])) % 10)) % 10
    full = norm[:12] + str(check)
    return ean_checksum_ok(full) and _in_catalog(full, catalog)


def _vote_track_barcodes(codes: list[str], *, min_votes: int = 2, catalog: set[str] | None = None) -> str:
    catalog = catalog or set()
    raw_votes: Counter[str] = Counter()
    for bc in codes:
        norm = normalize_barcode(bc)
        if len(norm) >= 12:
            raw_votes[norm] += 1
    if not raw_votes:
        return ""
    bc, n = raw_votes.most_common(1)[0]
    if n < min_votes:
        return ""
    if not _accept_barcode(bc, min_votes=min_votes, votes=n, catalog=catalog):
        return ""
    return bc


def infer_video(
    video: str,
    model,
    device,
    *,
    ocr_path: Path,
    out_path: Path | None = None,
    sharpest: bool = False,
    max_frames: int = 8,
    min_votes: int = 2,
    catalog: set[str] | None = None,
    group_fusion: bool = False,
    n_groups: int = 8,
) -> int:
    if group_fusion:
        from barcode_group_fusion import process_video as group_process
        return group_process(
            video, ocr_path, out_path,
            n_groups=n_groups, sharpest=sharpest, max_frames=max_frames,
            use_ml=True, catalog_path=ROOT / "db_hack.csv",
        )
    if not ocr_path.is_file():
        print(f"  skip {video}: no ocr at {ocr_path}")
        return 0
    df = pd.read_csv(ocr_path)
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    reader: _VideoReader | None = None
    vpath = find_video(f"{video}.mp4") or find_video(f"{video}/2.mp4")
    if vpath is not None:
        reader = _VideoReader(vpath)
        print(f"  video: {vpath.name} ({reader.w}x{reader.h})")

    updated = 0
    seen_track: set[int] = set()
    for tid in sorted(df["track_id"].dropna().astype(int).unique()):
        if tid in seen_track:
            continue
        sub = df[df["track_id"] == tid]
        if all(len(normalize_barcode(x)) >= 12 for x in sub.get("barcode_raw", [])):
            continue
        frames = _track_frame_rows(
            df, tid, sharpest=sharpest, reader=reader, max_frames=max_frames
        )
        preds: list[str] = []
        for _, row in frames.iterrows():
            cp = row.get("hires_crop") or row.get("crop_path")
            crop = _crop_from_row(row, reader)
            if crop is None and isinstance(cp, str) and cp.strip():
                p = ROOT / cp
                if p.is_file():
                    crop = cv2.imread(str(p))
            bc = _decode_crop(model, device, crop)
            if bc:
                preds.append(bc)
        bc = _vote_track_barcodes(preds, min_votes=min_votes, catalog=catalog)
        if len(normalize_barcode(bc)) < 12:
            continue
        mask = df["track_id"] == tid
        df.loc[mask, "barcode_raw"] = bc
        seen_track.add(tid)
        updated += 1
        print(f"  track {tid}: ML barcode {bc} ({len(preds)} frames)")

    if reader is not None:
        reader.close()

    dest = out_path or ocr_path
    df.to_csv(dest, index=False)
    print(f"  {video}: {updated} tracks → {dest.name}")
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ocr", type=Path, default=None, help="single ocr_qr csv (with --video)")
    ap.add_argument("--ocr-dir", type=Path, default=ROOT / "ml" / "output")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--sharpest", action="store_true", help="top Laplacian frames per track")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--min-votes", type=int, default=2, help="min agreeing frame predictions")
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--no-catalog", action="store_true", help="skip catalog membership check")
    ap.add_argument("--group-fusion", action="store_true",
                    help="7–10 digit groups fused across sharpest frames")
    ap.add_argument("--groups", type=int, default=8)
    args = ap.parse_args()
    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["26_12-20"])

    if args.group_fusion:
        from barcode_group_fusion import process_video as group_process
        use_ml = False
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _load_model(device)
            use_ml = True
        except SystemExit:
            print("No Slot-CNN checkpoint → OCR group fusion")
        total = 0
        for v in videos:
            ocr_path = args.ocr if args.ocr and len(videos) == 1 else args.ocr_dir / f"ocr_qr_{v}.csv"
            if args.out and len(videos) == 1:
                out_path = args.out
            elif args.in_place:
                out_path = ocr_path
            else:
                out_path = ocr_path.with_name(f"{ocr_path.stem}_groups.csv")
            total += group_process(
                v, ocr_path, out_path,
                n_groups=max(7, min(10, args.groups)),
                sharpest=args.sharpest, max_frames=max(args.max_frames, 12),
                use_ml=use_ml, catalog_path=args.catalog,
            )
        print(f"Total grouped barcodes: {total}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)
    catalog = set() if args.no_catalog else _load_catalog_barcodes(args.catalog)
    if catalog:
        print(f"Catalog: {len(catalog)} barcodes from {args.catalog.name}")
    total = 0
    for v in videos:
        if args.ocr and len(videos) == 1:
            ocr_path = args.ocr
        else:
            ocr_path = args.ocr_dir / f"ocr_qr_{v}.csv"
        out_path = None
        if args.out and len(videos) == 1:
            out_path = args.out
        elif not args.in_place:
            out_path = ocr_path.with_name(f"{ocr_path.stem}_ml.csv")
        total += infer_video(
            v, model, device,
            ocr_path=ocr_path,
            out_path=out_path,
            sharpest=args.sharpest,
            max_frames=args.max_frames,
            min_votes=args.min_votes,
            catalog=catalog,
        )
    print(f"Total ML barcodes: {total}")


if __name__ == "__main__":
    main()
