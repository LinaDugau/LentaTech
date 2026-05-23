"""Fine-tune YOLO v4."""
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


def make_combined_dataset(synth_dir: Path, gt_dir: Path, out_dir: Path):
    """Делает объединённый датасет: synth + gt → out_dir/.
    Файлы из gt именуются с префиксом gt_, чтобы не пересекаться с synth_.
    """
    import shutil
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for src_root, prefix in ((synth_dir, "synth"), (gt_dir, "gt")):
        for split in ("train", "val"):
            for img in (src_root / "images" / split).glob("*.jpg"):
                dst_img = out_dir / "images" / split / f"{prefix}_{img.name}"
                shutil.copy2(img, dst_img)
                lbl = src_root / "labels" / split / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, out_dir / "labels" / split / f"{prefix}_{img.stem}.txt")
                n_copied += 1
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: price_tag\n", encoding="utf-8")
    print(f"  combined dataset: {n_copied} images → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--model", default=str(
        ROOT / "ml" / "output" / "runs" / "pricetag_v2" / "weights" / "best.pt"))
    ap.add_argument("--name", default="pricetag_v4")
    ap.add_argument("--synth", default=str(ROOT / "ml" / "output" / "yolo_dataset_synth"))
    ap.add_argument("--gt", default=str(ROOT / "ml" / "output" / "yolo_dataset_gt"))
    ap.add_argument("--combined", default=str(ROOT / "ml" / "output" / "yolo_dataset_combined"))
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

    synth_dir = Path(args.synth)
    gt_dir = Path(args.gt)
    combined_dir = Path(args.combined)
    if not (synth_dir / "data.yaml").exists():
        print(f"❌ {synth_dir} не существует — сначала запусти synth_dataset.py")
        return
    if not (gt_dir / "data.yaml").exists():
        print(f"⚠️  {gt_dir} не существует — будем обучать только на synth")
        gt_dir = synth_dir  # дублирование на месте gt — безопасно
    print("Combining synth + gt datasets...")
    make_combined_dataset(synth_dir, gt_dir, combined_dir)

    device = pick_device()
    print(f"\nDevice: {device}")
    print(f"Warm start from: {args.model}")
    print(f"Saving to:       runs/{args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(combined_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "ml" / "output" / "runs"),
        name=args.name,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=8.0, translate=0.1, scale=0.4,
        fliplr=0.5, mosaic=0.5,
        cos_lr=True,
        patience=10,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
