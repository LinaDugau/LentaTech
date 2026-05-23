"""Sharpest-frame decode."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from video_resolve import find_video  # noqa: E402
from build_final_csv import choose_best_barcode, choose_best_qr  # noqa: E402
from whole_frame_codes2 import decode_frame, in_or_near, _propagate_tracks  # noqa: E402

MOTION_DOWNSCALE_W = 960
MOTION_MULT_DEFAULT = 1.8
TOP_K_DEFAULT = 2


def _downscale_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= MOTION_DOWNSCALE_W:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nh = int(h * MOTION_DOWNSCALE_W / w)
    small = cv2.resize(frame, (MOTION_DOWNSCALE_W, nh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _motion(prev: np.ndarray | None, curr: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return float(cv2.absdiff(prev, curr).mean())


def _tag_crop(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    h, w = frame.shape[:2]
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return None
    return frame[iy1:iy2, ix1:ix2]


def _blur_var(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _track_needs_decode(sub: pd.DataFrame) -> bool:
    qrs = [str(x) for x in sub.get("qr_raw", []) if str(x).strip() and str(x) != "nan"]
    bcs = [str(x) for x in sub.get("barcode_raw", []) if str(x).strip() and str(x) != "nan"]
    return not choose_best_qr(qrs) or not choose_best_barcode(bcs)


def _build_motion_map(cap: cv2.VideoCapture, frame_indices: list[int]) -> dict[int, float]:
    motion: dict[int, float] = {}
    prev_gray: np.ndarray | None = None
    prev_fi = -1
    for fi in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - 1))
        ok0, f0 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok1, f1 = cap.read()
        if not ok1 or f1 is None:
            continue
        g1 = _downscale_gray(f1)
        if fi - 1 == prev_fi and prev_gray is not None:
            motion[fi] = _motion(prev_gray, g1)
        elif ok0 and f0 is not None:
            motion[fi] = _motion(_downscale_gray(f0), g1)
        else:
            motion[fi] = 0.0
        prev_gray, prev_fi = g1, fi
    return motion


def _score_and_select(
    df: pd.DataFrame,
    cap: cv2.VideoCapture,
    motion_map: dict[int, float],
    motion_mult: float,
    top_k: int,
) -> tuple[list[int], float, float, dict[str, int]]:
    if "track_id" not in df.columns:
        return [], 0.0, 0.0, {"tracks": 0, "gated": 0, "fallback": 0}

    motion_vals = [v for v in motion_map.values() if v >= 0]
    motion_median = float(np.median(motion_vals)) if motion_vals else 0.0
    motion_thr = motion_mult * motion_median if motion_median > 0 else float("inf")

    rows_by_frame: dict[int, list[tuple[int, int, object]]] = defaultdict(list)
    stats = {"tracks": 0, "gated": 0, "fallback": 0}
    per_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    for tid, sub in df.groupby("track_id"):
        if not _track_needs_decode(sub):
            continue
        stats["tracks"] += 1
        for idx, row in sub.iterrows():
            fi = int(row["frame_idx_compute"])
            rows_by_frame[fi].append((idx, int(tid), row))

    for fi in tqdm(sorted(rows_by_frame), desc="score blur", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        mot = motion_map.get(fi, 0.0)
        for idx, tid, row in rows_by_frame[fi]:
            crop = _tag_crop(
                frame,
                float(row.x_min_orig), float(row.y_min_orig),
                float(row.x_max_orig), float(row.y_max_orig),
            )
            if crop is None:
                continue
            per_track[tid].append((idx, _blur_var(crop), mot))

    selected: list[int] = []
    for tid, scored in per_track.items():
        if not scored:
            continue
        gated = [t for t in scored if t[2] <= motion_thr]
        if gated:
            stats["gated"] += 1
            pick_from = gated
        else:
            stats["fallback"] += 1
            pick_from = scored
        pick_from.sort(key=lambda t: t[1], reverse=True)
        selected.extend(idx for idx, _, _ in pick_from[:top_k])

    return selected, motion_median, motion_thr, stats


def process(
    ocr_csv: Path,
    motion_mult: float = MOTION_MULT_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
) -> tuple[int, int]:
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return 0, 0
    for c in ("barcode_raw", "qr_raw"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
        else:
            df[c] = ""

    video_name = str(df.iloc[0]["video"])
    stem = ocr_csv.stem.replace("ocr_qr_", "")
    video = find_video(video_name)
    if video is None:
        print(f"  video not found: {video_name}")
        return 0, 0

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df["frame_idx_compute"] = (df["frame_ts_ms"] * fps / 1000).round().astype(int)
    unique_frames = sorted(df["frame_idx_compute"].unique())

    motion_map = _build_motion_map(cap, unique_frames)
    selected_rows, motion_median, motion_thr, stats = _score_and_select(
        df, cap, motion_map, motion_mult, top_k,
    )
    selected_set = set(selected_rows)
    decode_frames = sorted({int(df.at[i, "frame_idx_compute"]) for i in selected_rows})

    print(
        f"  motion median={motion_median:.2f} thr={motion_thr:.2f} "
        f"tracks={stats['tracks']} gated={stats['gated']} fallback={stats['fallback']}"
    )
    print(
        f"  selected {len(selected_rows)} rows / {len(decode_frames)} frames "
        f"(was {len(unique_frames)} unique OCR frames)"
    )

    n_bc_new = 0
    n_qr_new = 0

    for fi in tqdm(decode_frames, desc=f"{stem} decode"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        codes_b, codes_q = decode_frame(frame, w, h)
        rows_here = df[(df["frame_idx_compute"] == fi) & (df.index.isin(selected_set))]

        for idx, r in rows_here.iterrows():
            tb = (
                float(r.x_min_orig), float(r.y_min_orig),
                float(r.x_max_orig), float(r.y_max_orig),
            )
            if not str(df.at[idx, "barcode_raw"]).strip():
                for s, cb in codes_b:
                    if in_or_near(tb, cb):
                        df.at[idx, "barcode_raw"] = s
                        n_bc_new += 1
                        break
            if not str(df.at[idx, "qr_raw"]).strip():
                for s, cb in codes_q:
                    if in_or_near(tb, cb):
                        df.at[idx, "qr_raw"] = s
                        n_qr_new += 1
                        break

    cap.release()
    _propagate_tracks(df)
    df = df.drop(columns=["frame_idx_compute"], errors="ignore")
    df.to_csv(ocr_csv, index=False)

    qr_total = (df["qr_raw"].astype(str).str.len() > 5).sum()
    print(f"  decoded new: barcodes={n_bc_new}, QRs={n_qr_new}, qr_total={qr_total}/{len(df)}")
    return n_bc_new, n_qr_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--video", action="append")
    ap.add_argument("--motion-mult", type=float, default=MOTION_MULT_DEFAULT)
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    args = ap.parse_args()
    if args.files:
        files = [Path(p) for p in args.files]
    elif args.video:
        files = [ROOT / "ml" / "output" / f"ocr_qr_{v}.csv" for v in args.video]
    else:
        files = sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv"))
        files = [f for f in files if "_ml" not in f.stem and f.stem != "ocr_qr_input"]

    total_bc = total_qr = 0
    for f in files:
        print(f"\n=== {f.name} ===")
        bc, qr = process(f, motion_mult=args.motion_mult, top_k=args.top_k)
        total_bc += bc
        total_qr += qr
    print(f"\nTotal new: barcodes={total_bc}, QRs={total_qr}")


if __name__ == "__main__":
    main()
