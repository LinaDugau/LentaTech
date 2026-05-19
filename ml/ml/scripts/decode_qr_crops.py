"""День 13: декод QR/штрихкодов на расширенных QR-crop'ах."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "ml" / "weights" / "wechat_qr"

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
_WECHAT = None


def get_wechat():
    global _WECHAT
    if _WECHAT is None:
        try:
            _WECHAT = cv2.wechat_qrcode_WeChatQRCode(
                str(WEIGHTS / "detect.prototxt"),
                str(WEIGHTS / "detect.caffemodel"),
                str(WEIGHTS / "sr.prototxt"),
                str(WEIGHTS / "sr.caffemodel"))
        except Exception as e:
            print(f"  WeChat QR not loaded: {e}")
            _WECHAT = False
    return _WECHAT if _WECHAT is not False else None


def preprocess_variants(img: np.ndarray):
    yield "color", img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    yield "eq", cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    _, binary = cv2.threshold(eq, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "bin", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def upscale(img: np.ndarray, factor: float):
    if factor == 1.0:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def try_decode(img: np.ndarray) -> tuple[str, str]:
    bc, qr = "", ""
    wechat = get_wechat()
    if wechat is not None:
        try:
            results, _ = wechat.detectAndDecode(img)
            for r in results:
                if r and not qr:
                    qr = r
                    break
        except Exception:
            pass
    if qr and bc:
        return bc, qr
    for _, var in preprocess_variants(img):
        for factor in (1.0, 2.0):
            up = upscale(var, factor)
            for rot in (None, cv2.ROTATE_90_CLOCKWISE,
                        cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
                fr = cv2.rotate(up, rot) if rot is not None else up
                if HAS_PYZBAR and (not bc or not qr):
                    try:
                        for r in _zbar.decode(fr):
                            if not r.data:
                                continue
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
                if HAS_ZXING and (not bc or not qr):
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
    return bc, qr


def process(qr_summary_csv: Path, target_ocr_csv: Path):
    df_qr = pd.read_csv(qr_summary_csv)
    df_ocr = pd.read_csv(target_ocr_csv)
    for c in ("barcode_raw", "qr_raw"):
        if c in df_ocr.columns:
            df_ocr[c] = df_ocr[c].astype(object).where(df_ocr[c].notna(), "")

    key2idx_ocr: dict[tuple[int, int], int] = {}
    for idx, r in df_ocr.iterrows():
        rk = int(r["rank"]) if "rank" in df_ocr.columns and not pd.isna(r["rank"]) else 0
        key2idx_ocr[(int(r["track_id"]), rk)] = idx

    n_bc_new = 0
    n_qr_new = 0
    for r in tqdm(df_qr.itertuples(index=False), total=len(df_qr),
                  desc=qr_summary_csv.stem):
        qr_crop = getattr(r, "qr_crop", "")
        if not isinstance(qr_crop, str) or not qr_crop.strip():
            continue
        p = ROOT / qr_crop
        if not p.exists():
            continue
        rk = int(getattr(r, "rank", 0))
        key = (int(r.track_id), rk)
        if key not in key2idx_ocr:
            continue
        ocr_idx = key2idx_ocr[key]
        bc_existing = str(df_ocr.at[ocr_idx, "barcode_raw"]).strip()
        qr_existing = str(df_ocr.at[ocr_idx, "qr_raw"]).strip()
        if bc_existing and qr_existing:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        new_bc, new_qr = try_decode(img)
        if new_bc and not bc_existing:
            df_ocr.at[ocr_idx, "barcode_raw"] = new_bc
            n_bc_new += 1
        if new_qr and not qr_existing:
            df_ocr.at[ocr_idx, "qr_raw"] = new_qr
            n_qr_new += 1

    df_ocr.to_csv(target_ocr_csv, index=False)
    print(f"  decoded new: barcodes={n_bc_new}, QRs={n_qr_new}")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    out = ROOT / "ml" / "output"
    for qr_csv in sorted((out / "tracks").glob("track_summary_qr_*.csv")):
        stem = qr_csv.stem.replace("track_summary_qr_", "")
        target = out / f"ocr_qr_{stem}.csv"
        if not target.exists():
            print(f"  skip {stem}: no ocr csv")
            continue
        print(f"\n=== {stem} ===")
        process(qr_csv, target)


if __name__ == "__main__":
    main()
