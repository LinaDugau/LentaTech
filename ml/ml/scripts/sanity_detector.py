"""День 2: прямой sanity-check детектора (без трекера)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
WEIGHTS = ROOT / "ml" / "output" / "runs" / "pricetag_v1" / "weights" / "best.pt"

IOU_THR = 0.3


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def check_video(video: Path, gt_csv: Path, model: YOLO,
                imgsz: int = 960, max_long_side: int = 1920):
    gt = pd.read_csv(gt_csv)
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        gt[c] = gt[c].map(to_float)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = max_long_side / max(W0, H0)

    hits = 0
    total = len(gt)
    for _, row in gt.iterrows():
        ts_ms = row.frame_timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ok, frame = cap.read()
        if not ok:
            continue
        if scale < 1.0:
            frame_s = cv2.resize(frame, (int(W0 * scale), int(H0 * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            frame_s = frame
        res = model.predict(frame_s, conf=0.15, iou=0.5, imgsz=imgsz,
                            verbose=False)[0]
        gt_box = (row.x_min * scale, row.y_min * scale,
                  row.x_max * scale, row.y_max * scale)
        ok_match = False
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes.xyxy.cpu().numpy():
                if iou(b, gt_box) >= IOU_THR:
                    ok_match = True
                    break
        hits += int(ok_match)
    cap.release()
    return hits, total


def main():
    model = YOLO(str(WEIGHTS))
    print(f"weights: {WEIGHTS}")
    for sub in ("25_12-20", "26_12-20", "43_15"):
        video = DATA / sub / f"{sub}.mp4"
        gt = DATA / sub / f"{sub}.csv"
        hits, total = check_video(video, gt, model)
        recall = hits / total if total else 0.0
        print(f"{sub}: detector recall = {hits}/{total} = {recall:.3f}")


if __name__ == "__main__":
    sys.exit(main())
