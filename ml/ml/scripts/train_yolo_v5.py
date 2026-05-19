"""Обучение v5 — больше параметров, выше разрешение."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=6,
                    help="меньше из-за imgsz=1280 — больше памяти на картинку")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="warm start (v8s — 22 MB)")
    ap.add_argument("--name", default="pricetag_v5")
    ap.add_argument("--data", default=str(
        ROOT / "ml" / "output" / "yolo_dataset_combined" / "data.yaml"))
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    if not args.no_backup:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from backup_weights import backup_run
            existing = ROOT / "ml" / "output" / "runs" / args.name
            if existing.exists() and (existing / "weights" / "best.pt").exists():
                print(f"Auto-backup existing run: {args.name}")
                backup_run(args.name)
        except Exception as e:
            print(f"backup skipped: {e}")

    if not Path(args.data).exists():
        print(f"❌ Dataset не найден: {args.data}")
        print("Сначала запусти train_yolo_v4.py (он создаёт combined dataset)")
        return

    device = pick_device()
    print(f"\nDevice: {device}")
    print(f"Warm start from: {args.model}  (YOLOv8s ~22 MB)")
    print(f"imgsz: {args.imgsz}  multi_scale: True  epochs: {args.epochs}")
    print(f"Saving to: runs/{args.name}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "ml" / "output" / "runs"),
        name=args.name,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        degrees=10.0, translate=0.15, scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=0.7, mixup=0.1,
        cos_lr=True,
        patience=12,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
