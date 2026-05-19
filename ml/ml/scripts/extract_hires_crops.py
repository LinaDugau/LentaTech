"""День 3: извлечь high-res crop'ы из оригинального 4K видео."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
from ml.scripts.video_resolve import find_video  # noqa: E402


def process(summary_csv: Path, pad_pct: float = 0.06):
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    video_name = df.iloc[0]["video"]
    stem = summary_csv.stem.replace("track_summary_", "")
    video_path = find_video(video_name)
    if video_path is None:
        print(f"  video not found: {video_name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  source: {W}x{H} @ {fps:.1f}")

    out_dir = ROOT / "ml" / "output" / "tracks" / "best_frames_hires" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("frame_idx").reset_index(drop=True)
    cur_idx = -1
    cur_frame = None
    paths = []
    for r in tqdm(df_sorted.itertuples(index=False), total=len(df_sorted),
                  desc=stem):
        target = int(r.frame_idx)
        if target != cur_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, cur_frame = cap.read()
            if not ok:
                paths.append("")
                continue
            cur_idx = target
        x1, y1, x2, y2 = int(r.x_min_orig), int(r.y_min_orig), \
                         int(r.x_max_orig), int(r.y_max_orig)
        w, h = x2 - x1, y2 - y1
        px, py = int(w * pad_pct), int(h * pad_pct)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(W - 1, x2 + px); y2 = min(H - 1, y2 + py)
        if x2 <= x1 or y2 <= y1:
            paths.append("")
            continue
        crop = cur_frame[y1:y2, x1:x2]
        rank = int(getattr(r, "rank", 0))
        p = out_dir / f"track_{int(r.track_id):04d}_r{rank}.jpg"
        cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            paths.append(str(p.relative_to(ROOT)))
        except ValueError:
            paths.append(str(p))
    cap.release()

    df_sorted["hires_crop"] = paths
    out_csv = ROOT / "ml" / "output" / "tracks" / f"track_summary_hires_{stem}.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="*", default=None)
    ap.add_argument("--pad_pct", type=float, default=0.06)
    args = ap.parse_args()
    if args.summaries:
        files = [Path(p) for p in args.summaries]
    else:
        files = sorted((ROOT / "ml" / "output" / "tracks").glob(
            "track_summary_*.csv"))
        files = [f for f in files
                 if "_hires" not in f.name and "_qr" not in f.name]
    for f in files:
        print(f"\n=== {f.name} ===")
        process(f, args.pad_pct)


if __name__ == "__main__":
    main()
