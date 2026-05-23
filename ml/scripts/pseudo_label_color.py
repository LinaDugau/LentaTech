"""Pseudo-label по HSV."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "ml" / "output" / "frames_index.csv"
DATASET = ROOT / "ml" / "output" / "yolo_dataset"
VIS_DIR = ROOT / "ml" / "output" / "pseudo_vis"

RED_RANGES = [
    (np.array([0, 90, 70]),   np.array([10, 255, 255])),
    (np.array([170, 90, 70]), np.array([179, 255, 255])),
]
YELLOW_RANGE = (np.array([22, 150, 160]), np.array([32, 255, 255]))

MIN_W = 25
MIN_H = 25
MAX_W = 400
MAX_H = 450
MIN_AREA = 1200
MIN_FILL = 0.25  # доля цветной маски внутри bbox
MIN_ASPECT = 0.4  # h/w
MAX_ASPECT = 3.2


def color_mask(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = None
    for lo, hi in RED_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        red = m if red is None else cv2.bitwise_or(red, m)
    yellow = cv2.inRange(hsv, *YELLOW_RANGE)
    return red, yellow


def boxes_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                              iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < MIN_W or h < MIN_H:
            continue
        if w > MAX_W or h > MAX_H:
            continue
        if area < MIN_AREA:
            continue
        ar = h / max(w, 1)
        if ar < MIN_ASPECT or ar > MAX_ASPECT:
            continue
        fill = area / float(w * h)
        if fill < MIN_FILL:
            continue
        out.append((x, y, x + w, y + h, fill))
    return out


def nms(boxes: list[tuple], iou_thr: float = 0.3) -> list[tuple]:
    if not boxes:
        return []
    arr = np.array([[b[0], b[1], b[2], b[3], b[4]] for b in boxes],
                   dtype=np.float32)
    idxs = arr[:, 4].argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(arr[i, 0], arr[rest, 0])
        yy1 = np.maximum(arr[i, 1], arr[rest, 1])
        xx2 = np.minimum(arr[i, 2], arr[rest, 2])
        yy2 = np.minimum(arr[i, 3], arr[rest, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (arr[i, 2] - arr[i, 0]) * (arr[i, 3] - arr[i, 1])
        area_r = (arr[rest, 2] - arr[rest, 0]) * (arr[rest, 3] - arr[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idxs = rest[iou < iou_thr]
    return [boxes[i] for i in keep]


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
            (x2 - x1) / W, (y2 - y1) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--vis", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(INDEX)
    if args.limit:
        df = df.head(args.limit)
    print(f"Frames to process: {len(df)}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATASET / sub).mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    vis_indices = set(random.sample(range(len(df)), min(args.vis, len(df))))

    stats = {"frames": 0, "empty": 0, "boxes_red": 0, "boxes_yellow": 0}
    rows_out = []

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        img = cv2.imread(str(ROOT / row.saved_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m_red, m_yel = color_mask(hsv)

        red_boxes = nms(boxes_from_mask(m_red))
        yel_boxes = nms(boxes_from_mask(m_yel))
        stats["frames"] += 1
        stats["boxes_red"] += len(red_boxes)
        stats["boxes_yellow"] += len(yel_boxes)
        if not (red_boxes or yel_boxes):
            stats["empty"] += 1
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        vp = Path(row.video)
        stem = f"{vp.parent.name}__{vp.stem}_{row.frame_idx:06d}"
        img_dst = DATASET / "images" / split / f"{stem}.jpg"
        lbl_dst = DATASET / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with lbl_dst.open("w") as f:
            for (x1, y1, x2, y2, _) in red_boxes + yel_boxes:
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        rows_out.append({"video": row.video, "frame_idx": row.frame_idx,
                         "split": split, "n_red": len(red_boxes),
                         "n_yellow": len(yel_boxes)})

        if i in vis_indices:
            vis = img.copy()
            for (x1, y1, x2, y2, _) in red_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for (x1, y1, x2, y2, _) in yel_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite(str(VIS_DIR / f"{stem}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    yaml = DATASET / "data.yaml"
    yaml.write_text(
        f"path: {DATASET.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: price_tag\n",
        encoding="utf-8")

    summary = ROOT / "ml" / "output" / "pseudo_summary.csv"
    if rows_out:
        with summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  data.yaml: {yaml}")
    print(f"  vis dir:   {VIS_DIR}")


if __name__ == "__main__":
    main()
