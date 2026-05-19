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


def train_model(args):
    labels_path = args.data / "labels.csv"
    if not labels_path.is_file():
        raise SystemExit(f"No labels at {labels_path}. Run build_barcode_strip_dataset.py first.")

    rows = load_labels(labels_path)
    rng = np.random.default_rng(7)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_val = max(1, len(rows) // 5)
    val_idx = set(idx[:n_val].tolist())
    train_rows = [rows[i] for i in idx[n_val:]]
    val_rows = [rows[i] for i in idx[:n_val]]

    train_ds = StripDataset(
        train_rows, args.data, aug_per_real=args.aug_per_real, train=True,
        use_synth_font=args.synth_font, synth_per_real=args.synth_per_real,
    )
    val_ds = StripDataset(val_rows, args.data, aug_per_real=0, train=False, rng_seed=99)
    print(f"Train samples {len(train_ds)} | Val real {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlotCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    best_score = (-1.0, -1.0)  # (val_seq, val_digit)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_ok = tr_n = 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            logits = model(xs)
            loss = crit(logits, ys)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * xs.size(0)
            tr_ok += (logits.argmax(1) == ys).sum().item()
            tr_n += xs.size(0)

        model.eval()
        va_ok = va_n = 0
        seq_ok = seq_n = 0
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

        tr_acc = tr_ok / max(tr_n, 1)
        va_acc = va_ok / max(va_n, 1)
        va_seq = seq_ok / max(seq_n, 1)
        print(
            f"epoch {epoch:02d} loss={tr_loss / max(tr_n, 1):.4f} "
            f"train_digit={tr_acc:.3f} val_digit={va_acc:.3f} val_seq={va_seq:.3f}"
        )
        score = (va_seq, va_acc)
        if score >= best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_seq": va_seq,
                    "val_digit": va_acc,
                },
                MODEL_PATH,
            )

    print(
        f"Best val seq={best_score[0]:.3f} digit={best_score[1]:.3f} → {MODEL_PATH}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--synth-per-real", type=int, default=0)
    ap.add_argument("--aug-per-real", type=int, default=40)
    ap.add_argument("--synth-font", action="store_true")
    ap.add_argument("--holdout", default="", help="unused; random 20%% val split")
    args = ap.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
