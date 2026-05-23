"""Агрессивный decode штрихкодов и QR."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

try:
    from pyzbar import pyzbar as _zbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False

try:
    import zxingcpp as _zxing
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False

_BD = cv2.barcode.BarcodeDetector()
_QD = cv2.QRCodeDetector()


def variants(img: np.ndarray):
    """Yields (name, image) — несколько preprocessing-вариантов."""
    yield "color", img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    yield "eq", cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "bin", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def upscale(img: np.ndarray, factor: float):
    if factor == 1.0:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def rotations(img: np.ndarray):
    yield img
    yield cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    yield cv2.rotate(img, cv2.ROTATE_180)
    yield cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def try_decode(img: np.ndarray) -> tuple[str, str]:
    """Возвращает (barcode_value, qr_value). Пробует все декодеры на
    нескольких preprocessing-комбинациях. Stop at first success per type.
    """
    bc, qr = "", ""
    for _, var in variants(img):
        for factor in (2.0, 4.0):
            up = upscale(var, factor)
            for fr in rotations(up):
                if not bc and HAS_PYZBAR:
                    try:
                        for r in _zbar.decode(fr):
                            if r.data:
                                s = r.data.decode("utf-8", errors="ignore")
                                if r.type == "QRCODE":
                                    if not qr:
                                        qr = s
                                else:
                                    if not bc:
                                        bc = s
                                if bc and qr:
                                    return bc, qr
                    except Exception:
                        pass
                if not bc and HAS_ZXING:
                    try:
                        for r in _zxing.read_barcodes(fr):
                            if r.text:
                                if str(r.format).lower() == "qrcode":
                                    if not qr:
                                        qr = r.text
                                else:
                                    if not bc:
                                        bc = r.text
                                if bc and qr:
                                    return bc, qr
                    except Exception:
                        pass
                if not bc:
                    try:
                        ok, decoded, _, _ = _BD.detectAndDecodeWithType(fr)
                        if ok and decoded:
                            for s in decoded:
                                if s:
                                    bc = s
                                    break
                    except Exception:
                        pass
                if not qr:
                    try:
                        s, _, _ = _QD.detectAndDecode(fr)
                        if s:
                            qr = s
                    except Exception:
                        pass
                if bc and qr:
                    return bc, qr
    return bc, qr


def process(ocr_csv: Path):
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return
    if "hires_crop" not in df.columns:
        print(f"  no hires_crop column in {ocr_csv}")
        return

    for c in ("barcode_raw", "qr_raw"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")

    n_bc_new = 0
    n_qr_new = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=ocr_csv.stem):
        bc = str(row.barcode_raw) if not pd.isna(row.barcode_raw) else ""
        qr = str(row.qr_raw) if not pd.isna(row.qr_raw) else ""
        if bc.strip() and qr.strip():
            continue
        cp = row.hires_crop if isinstance(row.hires_crop, str) else None
        if not cp or not cp.strip():
            continue
        p = ROOT / cp
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        new_bc, new_qr = try_decode(img)
        if new_bc and not bc.strip():
            df.at[idx, "barcode_raw"] = new_bc
            n_bc_new += 1
        if new_qr and not qr.strip():
            df.at[idx, "qr_raw"] = new_qr
            n_qr_new += 1

    df.to_csv(ocr_csv, index=False)
    print(f"  decoded new: barcodes={n_bc_new}, QRs={n_qr_new}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()
    files = ([Path(p) for p in args.files] if args.files
             else sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv")))
    for f in files:
        print(f"\n=== {f.name} ===")
        process(f)


if __name__ == "__main__":
    main()
