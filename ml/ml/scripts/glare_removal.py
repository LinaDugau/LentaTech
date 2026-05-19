"""Glare / highlight removal — preprocessing для детекции и OCR."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _clahe_on_l(img: np.ndarray, clip: float = 1.5,
                 tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE на L канале LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _suppress_highlights(img: np.ndarray, v_thresh: int = 248,
                          s_thresh: int = 40,
                          min_blob_pct: float = 0.0008,
                          inpaint_radius: int = 3) -> np.ndarray:
    """Маска ТОЛЬКО ярких ПЯТЕН блика (большие связные области белого).

    Защита от шакаливания:
      - V > 248 (а не 235) — почти чистый белый
      - S < 40 — без насыщенности (значит блик, не белый объект с тиснением)
      - blob size > 0.08% площади — отсекаем мелкие точки/буквы
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > v_thresh) & (hsv[:, :, 1] < s_thresh)).astype(np.uint8) * 255
    if mask.sum() == 0:
        return img
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = int(h * w * min_blob_pct)
    big_mask = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            big_mask[labels == i] = 255
    if big_mask.sum() == 0:
        return img
    big_mask = cv2.dilate(big_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, big_mask, inpaint_radius, cv2.INPAINT_TELEA)


def _gamma(img: np.ndarray, gamma: float = 0.88) -> np.ndarray:
    """Гамма-коррекция — слегка темним всё."""
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def preprocess_for_detection(img: np.ndarray,
                              suppress_highlights: bool = True,
                              do_clahe: bool = True,
                              do_gamma: bool = False) -> np.ndarray:
    """Полный preprocessing для детектора и OCR.
    Идемпотентный (можно применять 1 раз перед всеми шагами).
    """
    out = img
    if suppress_highlights:
        out = _suppress_highlights(out)
    if do_clahe:
        out = _clahe_on_l(out)
    if do_gamma:
        out = _gamma(out, 0.88)
    return out


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Самая мягкая обработка — для OCR.
    НИКАКОГО inpaint (он может стирать буквы), только лёгкий CLAHE.
    """
    return _clahe_on_l(img, clip=1.2)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if src is None or not src.exists():
        print("Usage: python3 glare_removal.py <input.jpg> [output.jpg]")
        sys.exit(1)
    img = cv2.imread(str(src))
    out = preprocess_for_detection(img)
    if dst is None:
        dst = src.with_name(src.stem + "_clean" + src.suffix)
    cv2.imwrite(str(dst), out)
    print(f"Saved: {dst}")
