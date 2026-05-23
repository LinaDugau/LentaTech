"""EasyOCR + barcode/QR на crop."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import easyocr
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "ml" / "output" / "tracks"

_READER: easyocr.Reader | None = None
_BARCODE: cv2.barcode.BarcodeDetector | None = None
_QR: cv2.QRCodeDetector | None = None


def reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        kwargs = {}
        if model_dir := os.getenv("EASYOCR_MODEL_DIR"):
            kwargs["model_storage_directory"] = model_dir
        _READER = easyocr.Reader(["ru", "en"], gpu=False, verbose=False, **kwargs)
    return _READER


def rotations(img: np.ndarray) -> list[tuple[int, np.ndarray]]:
    return [
        (0, img),
        (90, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img, cv2.ROTATE_180)),
        (270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]


def _bbox_height(corners) -> float:
    ys = [p[1] for p in corners]
    return float(max(ys) - min(ys))


def best_ocr(img: np.ndarray) -> tuple[int, list[tuple[str, float, float]]]:
    """Возвращает (угол поворота, [(text, conf, bbox_height), ...]).

    bbox_height — высота bbox токена в пикселях ROTATED-кадра, годится
    как прокси размера шрифта.
    """
    best_angle = 0
    best_score = -1.0
    best_items: list[tuple[str, float, float]] = []
    for ang, rot in rotations(img):
        h, w = rot.shape[:2]
        scale_in = 1.0
        if max(h, w) < 200:
            scale_in = 200 / max(h, w)
            rot = cv2.resize(rot, (int(w * scale_in), int(h * scale_in)),
                             interpolation=cv2.INTER_CUBIC)
        results = reader().readtext(rot, detail=1, paragraph=False)
        items = [(t, float(c), _bbox_height(corners) / scale_in)
                 for (corners, t, c) in results]
        score = sum(c for _, c, _ in items)
        if score > best_score:
            best_score = score
            best_angle = ang
            best_items = items
    rotated = {0: img,
               90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
               180: cv2.rotate(img, cv2.ROTATE_180),
               270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}[best_angle]
    h, w = rotated.shape[:2]
    if h >= 30:
        bottom = rotated[int(h * 0.55):, :]
        bottom_up = cv2.resize(bottom, None, fx=2.0, fy=2.0,
                               interpolation=cv2.INTER_CUBIC)
        extra = reader().readtext(bottom_up, detail=1, paragraph=False,
                                  allowlist="0123456789-:./ ")
        for (corners, t, c) in extra:
            best_items.append((t, float(c), _bbox_height(corners) / 2.0))
    return best_angle, best_items


def decode_barcode(img: np.ndarray) -> str:
    """EAN/UPC через OpenCV BarcodeDetector. Пытаемся на всех ориентациях."""
    global _BARCODE
    if _BARCODE is None:
        _BARCODE = cv2.barcode.BarcodeDetector()
    for _, rot in rotations(img):
        ok, decoded, _, _ = _BARCODE.detectAndDecodeWithType(rot)
        if ok and decoded:
            for s in decoded:
                if s:
                    return s
    return ""


def decode_qr(img: np.ndarray) -> str:
    global _QR
    if _QR is None:
        _QR = cv2.QRCodeDetector()
    for _, rot in rotations(img):
        try:
            s, _, _ = _QR.detectAndDecode(rot)
        except cv2.error:
            continue
        if s:
            return s
    return ""


def process_summary(summary_csv: Path) -> Path:
    import pandas as pd
    df = pd.read_csv(summary_csv)
    rows = []
    use_hires = "hires_crop" in df.columns
    for r in tqdm(df.itertuples(index=False), total=len(df),
                  desc=summary_csv.stem):
        p = getattr(r, "hires_crop", "") if use_hires else r.crop_path
        if not p or (isinstance(p, float)):
            continue
        crop_path = ROOT / p
        img = cv2.imread(str(crop_path))
        if img is None:
            continue
        angle, items = best_ocr(img)
        texts = [it[0] for it in items]
        full_text = "\n".join(texts)
        big = img if max(img.shape[:2]) >= 500 else cv2.resize(
            img, None, fx=500 / max(img.shape[:2]),
            fy=500 / max(img.shape[:2]), interpolation=cv2.INTER_CUBIC)
        bc = decode_barcode(big)
        qr = decode_qr(big)
        rows.append({
            "video": r.video,
            "track_id": r.track_id,
            "frame_ts_ms": r.frame_ts_ms,
            "x_min_orig": r.x_min_orig, "y_min_orig": r.y_min_orig,
            "x_max_orig": r.x_max_orig, "y_max_orig": r.y_max_orig,
            "ocr_angle": angle,
            "ocr_text": full_text,
            "ocr_items_json": json.dumps(items, ensure_ascii=False),
            "barcode_raw": bc,
            "qr_raw": qr,
            "crop_path": r.crop_path,
            "hires_crop": p if use_hires else "",
        })

    base = summary_csv.stem.replace("track_summary_hires_", "").replace("track_summary_", "")
    out_csv = ROOT / "ml" / "output" / f"ocr_qr_{base}.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerows(rows)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="*", default=None,
                    help="track_summary_*.csv; по умолчанию все")
    args = ap.parse_args()
    if args.summaries:
        files = [Path(p) for p in args.summaries]
    else:
        files = sorted(TRACKS.glob("track_summary_*.csv"))
        files = [f for f in files if "_qr" not in f.name]
    for fp in files:
        out = process_summary(fp)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
