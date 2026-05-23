"""fine-tune YOLOv8n на псевдо-датасете ценников."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "ml" / "output" / "yolo_dataset_gt" / "data.yaml"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="pricetag_v1")
    ap.add_argument("--no_backup", action="store_true",
                    help="не делать бэкап существующих весов перед обучением")
    args = ap.parse_args()

    if not args.no_backup:
        try:
            from backup_weights import backup_run
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent))
            from backup_weights import backup_run
        existing = ROOT / "ml" / "output" / "runs" / args.name
        if existing.exists() and (existing / "weights" / "best.pt").exists():
            print(f"Auto-backup existing run: {args.name}")
            backup_run(args.name)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Data:   {DATA_YAML}")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "ml" / "output" / "runs"),
        name=args.name,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=10.0, translate=0.1, scale=0.4,
        fliplr=0.5, flipud=0.0,
        mosaic=0.5, mixup=0.0, copy_paste=0.0,
        patience=10,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
