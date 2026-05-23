"""Обучение slot-CNN на digit-strip (GT + синтетика)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import (  # noqa: E402
    N_DIGITS,
    STRIP_H,
    augment_strip,
    render_synthetic_strip,
    split_digit_slots,
)

DATA_DIR = ROOT / "ml" / "data" / "barcode_strips"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "barcode_digit_slots.pt"


class SlotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.fc = nn.Linear(96 * 4 * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class StripDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        root: Path,
        aug_per_real: int = 40,
        synth_per_real: int = 0,
        use_synth_font: bool = False,
        train: bool = True,
        rng_seed: int = 42,
    ):
        self.root = root
        self.train = train
        self.rng = np.random.default_rng(rng_seed + (0 if train else 1))
        self.samples: list[tuple[np.ndarray, str]] = []
        for row in rows:
            p = root / row["strip_path"]
            if not p.is_file():
                continue
            strip = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if strip is None:
                continue
            code = str(row["barcode"]).strip()
            if len(code) < 12:
                continue
            self.samples.append((strip, code))
            if train and aug_per_real > 0:
                for _ in range(aug_per_real):
                    self.samples.append((augment_strip(strip, self.rng), code))
            if train and use_synth_font and synth_per_real > 0:
                for _ in range(synth_per_real):
                    self.samples.append((render_synthetic_strip(code, self.rng), code))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        strip, code = self.samples[idx]
        code = code[:N_DIGITS]
        if len(code) < N_DIGITS:
            code = code.rjust(N_DIGITS, "0")
        slots = split_digit_slots(strip, n=N_DIGITS)
        xs = torch.from_numpy(slots.astype(np.float32) / 255.0).unsqueeze(1)  # (13,1,H,W)
        ys = torch.tensor([int(c) for c in code if c.isdigit()][:N_DIGITS], dtype=torch.long)
        if ys.numel() < N_DIGITS:
            ys = torch.nn.functional.pad(ys, (0, N_DIGITS - ys.numel()))
        return xs, ys


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)  # (B,13,1,H,W)
    ys = torch.stack([b[1] for b in batch], dim=0)  # (B,13)
    b, n, c, h, w = xs.shape
    xs = xs.view(b * n, c, h, w)
    ys = ys.view(b * n)
    return xs, ys


def load_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split_rows(rows: list[dict], holdout_video: str, val_frac: float, rng: np.random.Generator):
    real = [r for r in rows if r.get("source") in ("gt_video", "pred_crop")]
    pretrain = [r for r in rows if r.get("source") not in ("gt_video", "pred_crop")]

    if holdout_video:
        val_rows = [r for r in real if r.get("video") == holdout_video]
        train_real = [r for r in real if r.get("video") != holdout_video]
    else:
        idx = np.arange(len(real))
        rng.shuffle(idx)
        n_val = max(1, int(len(real) * val_frac))
        val_idx = set(idx[:n_val].tolist())
        train_real = [real[i] for i in idx[n_val:]]
        val_rows = [real[i] for i in idx[:n_val]]
    return train_real, val_rows, pretrain


def train_model(args):
    labels_path = args.data / "labels.csv"
    if not labels_path.is_file():
        raise SystemExit(f"No labels at {labels_path}. Run build_barcode_strip_dataset.py first.")

    rows = load_labels(labels_path)
    rng = np.random.default_rng(7)
    train_real, val_rows, pretrain_rows = _split_rows(
        rows, args.holdout_video, args.val_frac, rng
    )
    pretrain_data = args.pretrain_data
    if pretrain_data:
        pl = pretrain_data / "labels.csv"
        if pl.is_file():
            pretrain_rows = load_labels(pl)
            print(f"Extra pretrain labels: {len(pretrain_rows)} from {pretrain_data}")

    if getattr(args, "pretrain_max", 0) and len(pretrain_rows) > args.pretrain_max:
        rng.shuffle(pretrain_rows := list(pretrain_rows))
        pretrain_rows = pretrain_rows[: args.pretrain_max]
        print(f"  pretrain capped to {len(pretrain_rows)} rows")

    print(
        f"Real train {len(train_real)} | Val {len(val_rows)} | "
        f"Pretrain pool {len(pretrain_rows)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlotCNN().to(device)
    crit = nn.CrossEntropyLoss()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_score = (-1.0, -1.0)

    val_ds = StripDataset(val_rows, args.data, aug_per_real=0, train=False, rng_seed=99)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    def _eval_val() -> tuple[float, float]:
        model.eval()
        va_ok = va_n = seq_ok = seq_n = 0
        with torch.no_grad():
            for xs, ys in val_loader:
                xs, ys = xs.to(device), ys.to(device)
                logits = model(xs)
                va_ok += (logits.argmax(1) == ys).sum().item()
                va_n += xs.size(0)
                preds = logits.argmax(1).view(-1, N_DIGITS)
                targets = ys.view(-1, N_DIGITS)
                for p, t in zip(preds, targets):
                    seq_n += 1
                    if torch.equal(p, t):
                        seq_ok += 1
        return seq_ok / max(seq_n, 1), va_ok / max(va_n, 1)

    def _run_epochs(
        train_rows: list[dict],
        epochs: int,
        lr: float,
        aug: int,
        synth: int,
        tag: str,
        eval_each: bool = False,
    ):
        nonlocal model, best_score
        ds = StripDataset(
            train_rows, args.data, aug_per_real=aug, train=True,
            use_synth_font=args.synth_font, synth_per_real=synth,
        )
        if len(ds) == 0:
            print(f"  {tag}: no train samples, skip")
            return
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss = tr_ok = tr_n = 0
            for xs, ys in loader:
                xs, ys = xs.to(device), ys.to(device)
                opt.zero_grad()
                logits = model(xs)
                loss = crit(logits, ys)
                loss.backward()
                opt.step()
                tr_loss += float(loss) * xs.size(0)
                tr_ok += (logits.argmax(1) == ys).sum().item()
                tr_n += xs.size(0)
            msg = (
                f"  {tag} ep {epoch:02d} loss={tr_loss / max(tr_n, 1):.4f} "
                f"train_digit={tr_ok / max(tr_n, 1):.3f}"
            )
            if eval_each and len(val_ds):
                va_seq, va_acc = _eval_val()
                msg += f" val_digit={va_acc:.3f} val_seq={va_seq:.3f}"
                score = (va_seq, va_acc)
                if score >= best_score:
                    best_score = score
                    torch.save(
                        {"model": model.state_dict(), "val_seq": va_seq, "val_digit": va_acc},
                        MODEL_PATH,
                    )
            print(msg)

    if pretrain_rows and args.pretrain_epochs > 0:
        _run_epochs(
            pretrain_rows, args.pretrain_epochs, args.pretrain_lr,
            args.pretrain_aug, args.pretrain_synth, "pretrain",
        )

    finetune_rows = train_real if args.finetune_real_only else train_real + pretrain_rows
    _run_epochs(
        finetune_rows, args.epochs, args.lr,
        args.aug_per_real, args.synth_per_real, "finetune", eval_each=True,
    )

    if best_score[0] < 0:
        va_seq, va_acc = _eval_val()
        torch.save(
            {"model": model.state_dict(), "val_seq": va_seq, "val_digit": va_acc},
            MODEL_PATH,
        )
        best_score = (va_seq, va_acc)
    print(f"Saved → {MODEL_PATH} (val_seq={best_score[0]:.3f}, val_digit={best_score[1]:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--pretrain-data", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrain-lr", type=float, default=2e-3)
    ap.add_argument("--synth-per-real", type=int, default=0)
    ap.add_argument("--aug-per-real", type=int, default=60)
    ap.add_argument("--pretrain-aug", type=int, default=8)
    ap.add_argument("--pretrain-synth", type=int, default=2)
    ap.add_argument("--synth-font", action="store_true")
    ap.add_argument("--holdout-video", default="26_12-20")
    ap.add_argument("--pretrain-max", type=int, default=1500, help="cap pretrain rows")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--finetune-real-only", action="store_true", default=True)
    ap.add_argument("--no-finetune-real-only", action="store_false", dest="finetune_real_only")
    args = ap.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
