"""Визуализация GT-bbox на исходных кадрах — для проверки покрытия."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
OUT = ROOT / "ml" / "data" / "_gt_inspect"


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5,
                    help="кадров с каждого видео")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for video_dir in sorted(DATA.glob("*")):
        if not video_dir.is_dir() or video_dir.name == "Unlabeled":
            continue
        gt_csv = video_dir / f"{video_dir.name}.csv"
        video_path = video_dir / f"{video_dir.name}.mp4"
        if not (gt_csv.exists() and video_path.exists()):
            continue
        df = pd.read_csv(gt_csv)
        for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
            df[c] = df[c].map(to_float)

        unique_ts = df["frame_timestamp"].dropna().unique()[: args.max]

        cap = cv2.VideoCapture(str(video_path))
        for ts in unique_ts:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ok, frame = cap.read()
            if not ok:
                continue
            rows = df[df["frame_timestamp"] == ts]
            for _, r in rows.iterrows():
                x1, y1, x2, y2 = int(r.x_min), int(r.y_min), int(r.x_max), int(r.y_max)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                label = str(r.get("barcode", ""))[:15]
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            out_path = OUT / f"{video_dir.name}_ts{int(ts)}.jpg"
            h, w = frame.shape[:2]
            scale = 1080 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  saved: {out_path.name}  ({len(rows)} bbox)")
        cap.release()
    print(f"\n✅ Inspect images → {OUT}")


if __name__ == "__main__":
    main()
