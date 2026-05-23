"""EAN-13 из групп цифр с кадров."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import (  # noqa: E402
    N_DIGITS,
    build_digit_groups,
    ean_checksum_ok,
    extract_digit_strip,
    extract_group_strip,
    normalize_barcode,
    split_digit_slots,
)
from build_final_csv import fix_ocr_digits, _ean_checksum_ok  # noqa: E402
from extra_top_ocr import (  # noqa: E402
    _VideoReader, _blur_var, _crop_from_row, get_reader,
    _track_ids_from_final, _track_ids_from_gt,
)
from barcode_recovery_probe import decode_image  # noqa: E402
from partial_ean_recovery import _complete_candidates, _load_catalog  # noqa: E402
from video_resolve import find_video  # noqa: E402

VIDEOS = ("25_12-20", "26_12-20", "43_15")
Y_BANDS = ((0.52, 0.92), (0.48, 0.95))


@dataclass
class DigitVote:
    digit: str
    weight: float


@dataclass
class FrameObs:
    blur: float
    votes: dict[int, list[DigitVote]] = field(default_factory=dict)


def _preprocess_group_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    eq = cv2.resize(eq, (eq.shape[1] * 2, eq.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def _ocr_group_text(patch: np.ndarray) -> tuple[str, float]:
    if patch is None or patch.size == 0:
        return "", 0.0
    proc = _preprocess_group_patch(patch)
    try:
        results = get_reader().readtext(proc, detail=1, paragraph=False, allowlist="0123456789")
    except Exception:
        return "", 0.0
    digits = []
    conf_sum = 0.0
    for _, text, conf in results:
        d = re.sub(r"\D", "", str(text))
        if d:
            digits.append(d)
            conf_sum += float(conf)
    if not digits:
        return "", 0.0
    return "".join(digits), conf_sum / max(len(digits), 1)


def _decode_barcode_strip_votes(crop: np.ndarray, blur: float) -> Counter[str]:
    """zxing/cv2 на полосе штрихкода (не digit OCR)."""
    hits: Counter[str] = Counter()
    if crop is None or crop.size == 0:
        return hits
    h, w = crop.shape[:2]
    wgt = 1.0 + blur / 5000.0
    for y0, y1 in ((0.30, 0.78), (0.38, 0.88), (0.42, 0.95)):
        zone = crop[int(h * y0): int(h * y1), :]
        if zone.size == 0:
            continue
        for code in decode_image(zone, fast=False):
            norm = normalize_barcode(code)
            if len(norm) >= 12:
                hits[norm[:13] if len(norm) >= 13 else norm[:12]] += 2 * wgt
    return hits


def _ocr_observations(crop: np.ndarray, groups: list[tuple[int, int]], blur: float) -> FrameObs:
    obs = FrameObs(blur=blur)
    for y0, y1 in Y_BANDS:
        strip = extract_digit_strip(crop, y_start=y0, y_end=y1)
        if strip is None:
            continue
        for g_start, g_end in groups:
            patch = extract_group_strip(strip, g_start, g_end, upscale=4)
            text, conf = _ocr_group_text(patch)
            text = fix_ocr_digits(text)
            text = re.sub(r"\D", "", text)
            if not text:
                continue
            n_slots = g_end - g_start
            # align right if OCR shorter, left if longer
            if len(text) >= n_slots:
                chunk = text[:n_slots] if len(text) == n_slots else text[-n_slots:]
            else:
                chunk = text.rjust(n_slots, "?")
            w = conf * (1.0 + blur / 5000.0)
            for i, ch in enumerate(chunk):
                if ch.isdigit():
                    pos = g_start + i
                    obs.votes.setdefault(pos, []).append(DigitVote(ch, w))
    return obs


def _ml_observations(crop, model, device, groups, blur) -> FrameObs:
    import torch
    from train_barcode_strip import SlotCNN  # noqa: F401

    obs = FrameObs(blur=blur)
    for y0, y1 in Y_BANDS:
        strip = extract_digit_strip(crop, y_start=y0, y_end=y1)
        if strip is None:
            continue
        slots = split_digit_slots(strip, n=N_DIGITS)
        xs = torch.from_numpy(slots.astype(np.float32) / 255.0).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(xs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        for g_start, g_end in groups:
            for pos in range(g_start, g_end):
                d = str(int(pred[pos].item()))
                w = float(conf[pos].item()) * (1.0 + blur / 5000.0)
                obs.votes.setdefault(pos, []).append(DigitVote(d, w))
    return obs


def fuse_digit_observations(observations: list[FrameObs], min_weight: float = 0.15) -> list[str | None]:
    """Majority vote per digit position across frames/groups."""
    buckets: dict[int, Counter] = defaultdict(Counter)
    weights: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for obs in observations:
        for pos, votes in obs.votes.items():
            for v in votes:
                buckets[pos][v.digit] += v.weight
                weights[pos][v.digit] += v.weight
    result: list[str | None] = [None] * N_DIGITS
    for pos in range(N_DIGITS):
        if pos not in buckets:
            continue
        best_digit, best_w = "", 0.0
        for d, w in weights[pos].items():
            if w > best_w:
                best_w = w
                best_digit = d
        if best_digit and best_w >= min_weight:
            result[pos] = best_digit
    return result


def _digits_to_string(digits: list[str | None]) -> str:
    return "".join(d if d else "?" for d in digits)


def _fill_from_catalog(partial: str, catalog: set[str], prefix_idx: dict) -> str:
    known = partial.replace("?", "")
    if len(known) >= 10:
        for run in (partial.replace("?", ""), re.sub(r"\?+", "", partial)):
            if len(run) >= 10:
                cands = _complete_candidates(run, catalog, prefix_idx)
                if len(cands) == 1:
                    return cands[0]
    # try sliding windows of known digits
    s = partial.replace("?", "")
    for n in (12, 11, 10):
        if len(s) >= n:
            cands = _complete_candidates(s[:n], catalog, prefix_idx)
            if len(cands) == 1:
                return cands[0]
    return ""


def complete_ean(
    digits: list[str | None],
    *,
    catalog: set[str],
    prefix_idx: dict,
) -> str:
    partial = _digits_to_string(digits)
    filled = sum(1 for d in digits if d is not None)
    if filled < 6:
        return ""

    # checksum completion if 12 known
    known = [d for d in digits if d is not None]
    if filled >= 12:
        s = "".join(d if d else "0" for d in digits[:12])
        if "?" not in s[:12]:
            d = [int(c) for c in s[:12]]
            check = (10 - ((sum(d[0:11:2]) + 3 * sum(d[1:11:2])) % 10)) % 10
            full = s[:12] + str(check)
            if ean_checksum_ok(full) and (not catalog or full in catalog or full[:12] in catalog):
                return full

    cat_hit = _fill_from_catalog(partial, catalog, prefix_idx)
    if cat_hit:
        return cat_hit

    # brute: replace ? with votes from catalog prefix
    base = partial.replace("?", "")
    if len(base) >= 10 and catalog:
        for code in prefix_idx.get(base[:10], []):
            if ean_checksum_ok(code):
                return code
    return ""


def infer_track(
    frames_df: pd.DataFrame,
    reader: _VideoReader | None,
    *,
    groups: list[tuple[int, int]],
    model=None,
    device=None,
    max_frames: int = 16,
    sharpest: bool = True,
    catalog: set[str] | None = None,
    prefix_idx: dict | None = None,
) -> str:
    catalog = catalog or set()
    prefix_idx = prefix_idx or {}
    rows = frames_df
    if sharpest and reader is not None and len(rows) > max_frames:
        scored = []
        for idx, row in rows.iterrows():
            img = _crop_from_row(row, reader)
            scored.append((_blur_var(img) if img is not None else -1.0, idx))
        scored.sort(reverse=True)
        rows = frames_df.loc[[i for _, i in scored[:max_frames]]]

    observations: list[FrameObs] = []
    strip_hits: Counter[str] = Counter()
    for _, row in rows.iterrows():
        crop = _crop_from_row(row, reader)
        if crop is None:
            cp = row.get("hires_crop") or row.get("crop_path")
            if isinstance(cp, str) and cp.strip():
                p = ROOT / cp
                if p.is_file():
                    crop = cv2.imread(str(p))
        if crop is None or crop.size == 0:
            continue
        blur = _blur_var(crop)
        strip_hits += _decode_barcode_strip_votes(crop, blur)
        if model is not None and device is not None:
            obs = _ml_observations(crop, model, device, groups, blur)
        else:
            obs = _ocr_observations(crop, groups, blur)
        if obs.votes:
            observations.append(obs)

    if strip_hits:
        bc, n = strip_hits.most_common(1)[0]
        norm = normalize_barcode(bc)
        if len(norm) >= 12 and n >= 1.5:
            if not catalog or norm[:13] in catalog or norm[:12] in catalog:
                if len(norm) == 13 and ean_checksum_ok(norm):
                    return norm[:13]
                if len(norm) == 12:
                    return norm[:12]

    if not observations:
        return ""

    digits = fuse_digit_observations(observations)
    bc = complete_ean(digits, catalog=catalog, prefix_idx=prefix_idx)
    norm = normalize_barcode(bc)
    if len(norm) >= 12:
        if catalog and norm[:13] not in catalog and norm[:12] not in catalog:
            return ""
        return norm[:13] if len(norm) == 13 else norm[:12]
    return ""


def process_video(
    video: str,
    ocr_path: Path,
    out_path: Path | None = None,
    *,
    n_groups: int = 8,
    sharpest: bool = True,
    max_frames: int = 16,
    use_ml: bool = False,
    catalog_path: Path | None = None,
    track_ids: set[int] | None = None,
    only_missing: bool = True,
) -> int:
    if not ocr_path.is_file():
        print(f"  skip {video}: no {ocr_path}")
        return 0

    df = pd.read_csv(ocr_path)
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    groups = build_digit_groups(N_DIGITS, n_groups)
    print(f"  digit groups ({len(groups)}): {groups}")

    catalog, prefix_idx = _load_catalog(catalog_path or ROOT / "db_hack.csv")
    print(f"  catalog: {len(catalog)} EAN")

    model = device = None
    if use_ml:
        try:
            import torch
            from barcode_ml_infer import _load_model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _load_model(device)
            print("  ML model loaded")
        except SystemExit:
            print("  ML model missing → OCR groups only")
            use_ml = False

    reader: _VideoReader | None = None
    vpath = find_video(f"{video}.mp4") or find_video(f"{video}/2.mp4")
    if vpath is not None:
        reader = _VideoReader(vpath)

    updated = 0
    tids = sorted(df["track_id"].dropna().astype(int).unique())
    if track_ids:
        tids = [t for t in tids if t in track_ids]
        print(f"  filtered tracks: {len(tids)}")
    for i, tid in enumerate(tids):
        if i and i % 5 == 0:
            print(f"  progress: {i}/{len(tids)} tracks, {updated} barcodes found")
        sub = df[df["track_id"] == tid]
        if only_missing and all(len(normalize_barcode(x)) >= 12 for x in sub.get("barcode_raw", [])):
            continue
        bc = infer_track(
            sub, reader, groups=groups, model=model, device=device,
            max_frames=max_frames, sharpest=sharpest,
            catalog=catalog, prefix_idx=prefix_idx,
        )
        if len(normalize_barcode(bc)) < 12:
            continue
        df.loc[df["track_id"] == tid, "barcode_raw"] = bc
        updated += 1
        print(f"  track {tid}: grouped EAN {bc}")

    if reader is not None:
        reader.close()

    dest = out_path or ocr_path
    df.to_csv(dest, index=False)
    print(f"  {video}: {updated} tracks updated → {dest.name}")
    return updated


def main():
    ap = argparse.ArgumentParser(description="EAN from 7–10 digit groups across video frames")
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ocr", type=Path, default=None)
    ap.add_argument("--ocr-dir", type=Path, default=ROOT / "ml" / "output")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--groups", type=int, default=8, help="7–10 overlapping digit groups")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--sharpest", action="store_true", default=True)
    ap.add_argument("--no-sharpest", action="store_false", dest="sharpest")
    ap.add_argument("--use-ml", action="store_true", help="Slot-CNN per digit + group fusion")
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--final", type=Path, default=None, help="restrict to tracks matched to final rows")
    ap.add_argument("--gt", type=Path, default=None, help="restrict to tracks matched to GT rows")
    ap.add_argument("--only-missing", action="store_true", default=True,
                    help="skip tracks that already have barcode_raw (default)")
    ap.add_argument("--all-tracks", action="store_true", help="process even if barcode_raw present")
    args = ap.parse_args()

    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["26_12-20"])
    total = 0
    for v in videos:
        ocr_path = args.ocr if args.ocr and len(videos) == 1 else args.ocr_dir / f"ocr_qr_{v}.csv"
        if args.out and len(videos) == 1:
            out_path = args.out
        elif args.in_place:
            out_path = ocr_path
        else:
            out_path = ocr_path.with_name(f"{ocr_path.stem}_groups.csv")
        track_ids = None
        if args.gt and args.gt.is_file():
            track_ids = _track_ids_from_gt(args.gt, ocr_path)
            print(f"  --gt: {len(track_ids)} track_ids for {v}")
        elif args.final and args.final.is_file():
            track_ids = _track_ids_from_final(args.final, ocr_path)
            print(f"  --final: {len(track_ids)} track_ids for {v}")
        only_missing = args.only_missing and not args.all_tracks
        total += process_video(
            v, ocr_path, out_path,
            n_groups=max(7, min(10, args.groups)),
            sharpest=args.sharpest,
            max_frames=args.max_frames,
            use_ml=args.use_ml,
            catalog_path=args.catalog,
            track_ids=track_ids,
            only_missing=only_missing,
        )
    print(f"Total grouped barcodes: {total}")


if __name__ == "__main__":
    main()
