"""Pseudo-label YOLO-World."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "ml" / "output" / "frames_index.csv"
DATASET = ROOT / "ml" / "output" / "yolo_dataset"
VIS_DIR = ROOT / "ml" / "output" / "pseudo_vis"

PROMPTS = ["price tag", "price label", "paper price sign", "shelf label"]
CONF = 0.05
IOU_NMS = 0.5

MIN_SIDE_PX = 25
MAX_SIDE_PX = 350
MIN_ASPECT = 0.4  # h/w
MAX_ASPECT = 3.0


def filter_box(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return False
    if w > MAX_SIDE_PX or h > MAX_SIDE_PX:
        return False
    ar = h / max(w, 1)
    if ar < MIN_ASPECT or ar > MAX_ASPECT:
        return False
    return True


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt",
                    help="YOLO-World checkpoint (auto-downloaded)")
    ap.add_argument("--limit", type=int, default=0, help="0 = все кадры")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--vis", type=int, default=20,
                    help="Сохранить N кадров с нарисованными bbox для визуальной проверки")
    args = ap.parse_args()

    df = pd.read_csv(INDEX)
    if args.limit:
        df = df.head(args.limit)
    print(f"Frames to process: {len(df)}")

    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / "val").mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model = YOLO(args.model)
    model.set_classes(PROMPTS)

    random.seed(42)
    vis_indices = set(random.sample(range(len(df)), min(args.vis, len(df))))

    stats = {"frames": 0, "empty": 0, "boxes_raw": 0, "boxes_kept": 0}
    rows_out = []

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        img_path = ROOT / row.saved_path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        res = model.predict(img, conf=CONF, iou=IOU_NMS, verbose=False)[0]

        kept = []
        for b in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b
            stats["boxes_raw"] += 1
            if filter_box(x1, y1, x2, y2):
                kept.append((x1, y1, x2, y2))
        stats["boxes_kept"] += len(kept)
        stats["frames"] += 1
        if not kept:
            stats["empty"] += 1
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        stem = f"{Path(row.video).stem}_{row.frame_idx:06d}"
        img_dst = DATASET / "images" / split / f"{stem}.jpg"
        lbl_dst = DATASET / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with lbl_dst.open("w") as f:
            for (x1, y1, x2, y2) in kept:
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        rows_out.append({"video": row.video, "frame_idx": row.frame_idx,
                         "split": split, "n_boxes": len(kept)})

        if i in vis_indices:
            vis = img.copy()
            for (x1, y1, x2, y2) in kept:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
            cv2.imwrite(str(VIS_DIR / f"{stem}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: price_tag\n", encoding="utf-8"
    )

    pseudo_csv = ROOT / "ml" / "output" / "pseudo_summary.csv"
    if rows_out:
        with pseudo_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  vis dir:   {VIS_DIR}")


if __name__ == "__main__":
    main()
