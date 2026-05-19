"""Общие утилиты для digit-strip barcode ML."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

STRIP_H = 32
STRIP_W = 208  # 13 * 16 px per digit
N_DIGITS = 13


def ean_checksum_ok(code: str) -> bool:
    if not re.fullmatch(r"\d{13}", code):
        return False
    d = [int(c) for c in code]
    check = (10 - ((sum(d[0:12:2]) + 3 * sum(d[1:12:2])) % 10)) % 10
    return check == d[-1]


def normalize_barcode(code) -> str:
    if code is None:
        return ""
    if isinstance(code, float):
        if code != code:  # NaN
            return ""
        s = f"{code:.0f}"
    else:
        s = str(code)
    s = re.sub(r"\D", "", s)
    if len(s) == 12:
        return s
    if len(s) == 13 and ean_checksum_ok(s):
        return s
    if len(s) == 13:
        return s[:12] if not ean_checksum_ok(s) else s
    return ""


def extract_digit_strip(
    img: np.ndarray,
    y_start: float = 0.52,
    y_end: float = 0.92,
    pad_x: float = 0.02,
) -> np.ndarray | None:
    """Нижняя зона ценника под штрихкодом (цифры EAN)."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    y0 = max(0, int(h * y_start))
    y1 = min(h, int(h * y_end))
    x0 = max(0, int(w * pad_x))
    x1 = min(w, int(w * (1.0 - pad_x)))
    if y1 - y0 < 8 or x1 - x0 < 40:
        return None
    strip = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    gray = cv2.resize(gray, (STRIP_W, STRIP_H), interpolation=cv2.INTER_AREA)
    return gray


def split_digit_slots(strip: np.ndarray, n: int = N_DIGITS) -> np.ndarray:
    """(n, STRIP_H, slot_w) grayscale patches."""
    h, w = strip.shape[:2]
    slot_w = max(1, w // n)
    slots = []
    for i in range(n):
        x0 = i * slot_w
        x1 = w if i == n - 1 else (i + 1) * slot_w
        patch = strip[:, x0:x1]
        patch = cv2.resize(patch, (16, STRIP_H), interpolation=cv2.INTER_AREA)
        slots.append(patch)
    return np.stack(slots, axis=0)


def augment_strip(strip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = strip.copy()
    if rng.random() < 0.7:
        alpha = rng.uniform(0.7, 1.4)
        beta = rng.integers(-35, 35)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.3, 1.2))
    if rng.random() < 0.4:
        noise = rng.normal(0, rng.uniform(3, 12), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        angle = rng.uniform(-2.5, 2.5)
        m = cv2.getRotationMatrix2D((STRIP_W / 2, STRIP_H / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (STRIP_W, STRIP_H), borderMode=cv2.BORDER_REPLICATE)
    return out


def render_synthetic_strip(
    barcode: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Синтетическая полоска цифр под штрихкодом."""
    rng = rng or np.random.default_rng()
    code = re.sub(r"\D", "", barcode)
    n = len(code)
    w = STRIP_W if n >= 13 else int(STRIP_W * n / 13)
    img = np.full((STRIP_H, w), 245, dtype=np.uint8)
    slot_w = w // max(n, 1)
    font = rng.choice(
        [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
    )
    scale = rng.uniform(0.42, 0.62)
    thick = rng.choice([1, 2])
    for i, ch in enumerate(code):
        x = i * slot_w + rng.integers(0, max(1, slot_w // 4))
        y = STRIP_H - rng.integers(3, 8)
        cv2.putText(img, ch, (x, y), font, scale, 0, thick, cv2.LINE_AA)
    if w != STRIP_W:
        img = cv2.resize(img, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
    return augment_strip(img, rng)
