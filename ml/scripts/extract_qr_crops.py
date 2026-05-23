"""QR-crop с расширенным pad."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
from ml.scripts.video_resolve import find_video  # noqa: E402


def process(summary_csv: Path, pad_x: float = 0.40,
            pad_y_top: float = 1.20, pad_y_bot: float = 0.40):
    """pad_y_top=1.20 → расширение ВВЕРХ на 120% высоты bbox (QR обычно сверху).
    pad_y_bot — небольшое расширение вниз на случай нестандартной ориентации.
    """
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    video_name = df.iloc[0]["video"]
    stem = summary_csv.stem.replace("track_summary_hires_", "").replace("track_summary_", "")
    video_path = find_video(video_name)
    if video_path is None:
        print(f"  video not found: {video_name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = ROOT / "ml" / "output" / "tracks" / "best_frames_qr" / stem
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
        is_vertical = h >= w
        if is_vertical:
            px = int(w * pad_x)
            py_top = int(h * pad_y_top)
            py_bot = int(h * pad_y_bot)
        else:
            px = int(w * max(pad_x, 0.6))
            py_top = int(h * pad_x)
            py_bot = int(h * pad_x)
        x1e = max(0, x1 - px)
        y1e = max(0, y1 - py_top)
        x2e = min(W - 1, x2 + px)
        y2e = min(H - 1, y2 + py_bot)
        if x2e <= x1e or y2e <= y1e:
            paths.append("")
            continue
        crop = cur_frame[y1e:y2e, x1e:x2e]
        rank = int(getattr(r, "rank", 0))
        p = out_dir / f"track_{int(r.track_id):04d}_r{rank}.jpg"
        cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            paths.append(str(p.relative_to(ROOT)))
        except ValueError:
            paths.append(str(p))
    cap.release()

    df_sorted["qr_crop"] = paths
    out_csv = ROOT / "ml" / "output" / "tracks" / f"track_summary_qr_{stem}.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="*", default=None)
    ap.add_argument("--pad_y_top", type=float, default=1.20)
    ap.add_argument("--pad_y_bot", type=float, default=0.40)
    args = ap.parse_args()
    files = ([Path(p) for p in args.summaries] if args.summaries
             else sorted((ROOT / "ml" / "output" / "tracks").glob(
                 "track_summary_hires_*.csv")))
    for f in files:
        print(f"\n=== {f.name} ===")
        process(f, pad_y_top=args.pad_y_top, pad_y_bot=args.pad_y_bot)


if __name__ == "__main__":
    main()
