"""Датасет с РАСШИРЕННЫМИ GT-bbox, покрывающими весь ценник (red+white+QR)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ml" / "output" / "yolo_dataset_full_tag"))
    ap.add_argument("--side", choices=["right", "left", "down", "up", "both_x", "both_y", "all"],
                    default="all",
                    help="в какую сторону расширять (QR обычно сбоку или сверху)")
    ap.add_argument("--extend", type=float, default=0.5,
                    help="доля высоты/ширины bbox для расширения")
    ap.add_argument("--neighbor_frames", type=int, default=3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--max_long_side", type=int, default=1920)
    args = ap.parse_args()

    OUT = Path(args.out)
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / s).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    stats = {"frames": 0, "bbox": 0, "videos": 0}

    for video_dir in sorted(DATA.glob("*")):
        if not video_dir.is_dir() or video_dir.name == "Unlabeled":
            continue
        video_path = video_dir / f"{video_dir.name}.mp4"
        gt_csv = video_dir / f"{video_dir.name}.csv"
        if not (video_path.exists() and gt_csv.exists()):
            continue
        stats["videos"] += 1

        gt = pd.read_csv(gt_csv)
        for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
            gt[c] = gt[c].map(to_float)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = args.max_long_side / max(W0, H0)
        Wn, Hn = int(W0 * scale), int(H0 * scale)

        gt["frame_idx_ref"] = (gt["frame_timestamp"] * fps / 1000).round().astype(int)

        frames_to_export: dict[int, list[tuple[float, float, float, float]]] = {}
        for _, r in gt.iterrows():
            ref = int(r.frame_idx_ref)
            x1, y1, x2, y2 = r.x_min, r.y_min, r.x_max, r.y_max
            w, h = x2 - x1, y2 - y1
            ex_w_right = w * args.extend if args.side in ("right", "both_x", "all") else 0
            ex_w_left  = w * args.extend if args.side in ("left",  "both_x", "all") else 0
            ex_h_down  = h * args.extend if args.side in ("down",  "both_y", "all") else 0
            ex_h_up    = h * args.extend if args.side in ("up",    "both_y", "all") else 0
            x1e = max(0, x1 - ex_w_left)
            x2e = min(W0 - 1, x2 + ex_w_right)
            y1e = max(0, y1 - ex_h_up)
            y2e = min(H0 - 1, y2 + ex_h_down)
            for d in range(-args.neighbor_frames, args.neighbor_frames + 1):
                fi = ref + d
                if fi < 0:
                    continue
                frames_to_export.setdefault(fi, []).append((x1e, y1e, x2e, y2e))

        for fi in tqdm(sorted(frames_to_export.keys()),
                       desc=video_dir.name, leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_s = cv2.resize(frame, (Wn, Hn), interpolation=cv2.INTER_AREA)
            split = "val" if random.random() < args.val_ratio else "train"
            stem = f"{video_dir.name}_{fi:06d}"
            img_path = OUT / "images" / split / f"{stem}.jpg"
            lbl_path = OUT / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(img_path), frame_s, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with lbl_path.open("w") as f:
                for (x1, y1, x2, y2) in frames_to_export[fi]:
                    x1s, y1s = x1 * scale, y1 * scale
                    x2s, y2s = x2 * scale, y2 * scale
                    cx = (x1s + x2s) / 2 / Wn
                    cy = (y1s + y2s) / 2 / Hn
                    bw = (x2s - x1s) / Wn
                    bh = (y2s - y1s) / Hn
                    if bw <= 0 or bh <= 0:
                        continue
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    stats["bbox"] += 1
            stats["frames"] += 1
        cap.release()

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: price_tag\n", encoding="utf-8")
    print("\nDONE")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  side={args.side}  extend={args.extend}")
    print(f"  data.yaml: {OUT}")


if __name__ == "__main__":
    main()
