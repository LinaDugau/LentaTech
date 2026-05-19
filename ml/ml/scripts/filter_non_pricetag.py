"""Post-filter best-frame crops, выкидываем «не-ценники» (упаковка/полки/мусор)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def paper_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s < 60) & (v > 170)
    return float(mask.mean())


def sharpness(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def judge(crop_path: Path, ar_min: float, ar_max: float,
          paper_min: float, sharp_min: float) -> tuple[bool, str, dict]:
    if not crop_path.exists():
        return False, "missing", {}
    img = cv2.imread(str(crop_path))
    if img is None or img.size == 0:
        return False, "unreadable", {}
    h, w = img.shape[:2]
    ar = h / max(w, 1)
    pr = paper_ratio(img)
    sh = sharpness(img)
    stats = {"ar": round(ar, 2), "paper": round(pr, 3), "sharp": round(sh, 1)}
    if ar < ar_min or ar > ar_max:
        return False, f"ar({ar:.2f})", stats
    if pr < paper_min:
        return False, f"paper({pr:.2f})", stats
    if sh < sharp_min:
        return False, f"sharp({sh:.0f})", stats
    return True, "ok", stats


def process_video(stem: str, args) -> dict:
    ts_path = ROOT / "ml" / "output" / "tracks" / f"track_summary_{stem}.csv"
    ocr_path = ROOT / "ml" / "output" / f"ocr_qr_{stem}.csv"
    if not ts_path.exists():
        print(f"  [{stem}] no track_summary, skip")
        return {}

    ts = pd.read_csv(ts_path)
    n_tracks_in = ts["track_id"].nunique()
    n_rows_in = len(ts)

    rank0 = ts[ts["rank"] == 0].copy()
    keep_tids: set[int] = set()
    drop_reasons: dict[str, int] = {}
    sample_drops: list[tuple[int, str, dict]] = []

    for _, row in rank0.iterrows():
        cp = ROOT / str(row["crop_path"])
        ok, reason, stats = judge(cp, args.ar_min, args.ar_max,
                                  args.paper_min, args.sharp_min)
        if ok:
            keep_tids.add(int(row["track_id"]))
        else:
            drop_reasons[reason.split("(")[0]] = drop_reasons.get(reason.split("(")[0], 0) + 1
            if len(sample_drops) < 5:
                sample_drops.append((int(row["track_id"]), reason, stats))

    ts_keep = ts[ts["track_id"].isin(keep_tids)].copy()

    if not args.dry_run:
        bak = ts_path.with_suffix(".csv.bak")
        if not bak.exists():
            shutil.copy(ts_path, bak)
        ts_keep.to_csv(ts_path, index=False)

    ocr_stats = ""
    if ocr_path.exists():
        ocr = pd.read_csv(ocr_path)
        n_ocr_in = len(ocr)
        ocr_keep = ocr[ocr["track_id"].isin(keep_tids)].copy()
        if not args.dry_run:
            bak = ocr_path.with_suffix(".csv.bak")
            if not bak.exists():
                shutil.copy(ocr_path, bak)
            ocr_keep.to_csv(ocr_path, index=False)
        ocr_stats = f", ocr {n_ocr_in}→{len(ocr_keep)}"

    print(f"  [{stem}] tracks {n_tracks_in}→{len(keep_tids)}, "
          f"rows {n_rows_in}→{len(ts_keep)}{ocr_stats}")
    print(f"         drop reasons: {drop_reasons}")
    if sample_drops:
        print(f"         sample drops:")
        for tid, reason, stats in sample_drops:
            print(f"           track {tid:>4d}  {reason:<14s}  {stats}")
    return {"tracks_in": n_tracks_in, "tracks_kept": len(keep_tids),
            "drops": drop_reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+",
                    default=["25_12-20", "26_12-20", "43_15"])
    ap.add_argument("--ar_min", type=float, default=0.35,
                    help="min aspect ratio h/w (default 0.35 — drop узкие полосы полок)")
    ap.add_argument("--ar_max", type=float, default=2.8,
                    help="max aspect ratio h/w (default 2.8 — drop вытянутые корешки)")
    ap.add_argument("--paper_min", type=float, default=0.10,
                    help="min доля 'бумажных' пикселей S<60 V>170 (default 0.10)")
    ap.add_argument("--sharp_min", type=float, default=40.0,
                    help="min Laplacian.var (default 40 — drop полностью размытые)")
    ap.add_argument("--dry_run", action="store_true",
                    help="не писать, только показать статистику")
    args = ap.parse_args()

    print(f"Thresholds: AR[{args.ar_min},{args.ar_max}] "
          f"paper≥{args.paper_min} sharp≥{args.sharp_min} "
          f"{'(dry-run)' if args.dry_run else ''}")
    print()
    for v in args.videos:
        process_video(v, args)
        print()


if __name__ == "__main__":
    sys.exit(main())
