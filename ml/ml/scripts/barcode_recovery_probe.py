"""Быстрый probe: стоит ли вкладываться в barcode recovery?"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from evaluate_official import IOU_THR, TS_TOL_MS, match_pairs, _to_float  # noqa: E402
from video_resolve import find_video  # noqa: E402

try:
    import zxingcpp as _zxing
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False

_BD = cv2.barcode.BarcodeDetector()


def _ean13_ok(code: str) -> bool:
    if not re.fullmatch(r"\d{13}", code):
        return False
    d = [int(c) for c in code]
    check = (10 - ((sum(d[0:12:2]) + 3 * sum(d[1:12:2])) % 10)) % 10
    return check == d[-1]


def _zones(img):
    h, w = img.shape[:2]
    yield img
    for y1 in (0.45, 0.55, 0.65):
        c = img[int(h * y1):, :]
        if c.size:
            yield c


def _preprocess(gray, scale: float = 3.0):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 2.0, blur, -1.0, 0)
    for base in (sharp, clahe):
        im = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if max(im.shape) > 2200:
            f = 2200 / max(im.shape)
            im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        yield im


def decode_image(img, fast: bool = False) -> set[str]:
    found: set[str] = set()
    if img is None or img.size == 0:
        return found
    rots = [None] if fast else [None, cv2.ROTATE_90_CLOCKWISE,
                                cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    scales = (2.5,) if fast else (2.5, 4.0)

    for zone in _zones(img):
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if zone.ndim == 3 else zone
        for scale in scales:
            for proc in _preprocess(gray, scale=scale):
                bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                for rot in rots:
                    fr = cv2.rotate(bgr, rot) if rot is not None else bgr
                    if HAS_ZXING:
                        try:
                            for r in _zxing.read_barcodes(fr):
                                s = str(r.text or "").strip()
                                if re.fullmatch(r"\d{12,14}", s):
                                    found.add(s)
                        except Exception:
                            pass
                    try:
                        ok, decoded, _, _ = _BD.detectAndDecodeWithType(fr)
                        if ok and decoded:
                            for s in decoded:
                                s = str(s or "").strip()
                                if re.fullmatch(r"\d{12,14}", s):
                                    found.add(s)
                    except Exception:
                        pass
                    if found and fast:
                        return found
    return found


def _resolve(p: str) -> Path | None:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.is_file() else None


def _gt_box(g) -> tuple[float, float, float, float]:
    return (_to_float(g["x_min"]), _to_float(g["y_min"]),
            _to_float(g["x_max"]), _to_float(g["y_max"]))


def _expand_for_barcode(x1, y1, x2, y2, W, H, pad_x=0.35, pad_down=0.80):
    w, h = x2 - x1, y2 - y1
    return (max(0, int(x1 - w * pad_x)),
            max(0, int(y1 - h * 0.15)),
            min(W - 1, int(x2 + w * pad_x)),
            min(H - 1, int(y2 + h * pad_down)))


def _ocr_track_for_pred(ocr: pd.DataFrame, pred_row) -> int | None:
    ts = float(pred_row["frame_timestamp"])
    x0, y0 = float(pred_row["x_min"]), float(pred_row["y_min"])
    cand = ocr[
        (ocr["frame_ts_ms"].astype(float).sub(ts).abs() < 10)
        & (ocr["x_min_orig"].astype(float).sub(x0).abs() < 8)
        & (ocr["y_min_orig"].astype(float).sub(y0).abs() < 8)
    ]
    if cand.empty:
        cand = ocr[ocr["frame_ts_ms"].astype(float).sub(ts).abs() < TS_TOL_MS]
        if not cand.empty:
            cand = cand.assign(_d=(
                (cand["x_min_orig"].astype(float) - x0).abs()
                + (cand["y_min_orig"].astype(float) - y0).abs()
            )).sort_values("_d")
    return int(cand.iloc[0]["track_id"]) if not cand.empty else None


def _video_crop(cap, ts_ms: float, box, W, H) -> cv2.Mat | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    x1, y1, x2, y2 = _expand_for_barcode(*box, W, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _status(gtbc: str, hits: Counter[str]) -> str:
    codes = set(hits)
    if gtbc in codes:
        return "EXACT"
    if any(c.startswith(gtbc[:12]) for c in codes):
        return "PREFIX12"
    if codes:
        return f"ANY({len(codes)})"
    return "miss"


def probe(video: str, limit: int = 8, ranks: int = 2, fast: bool = False) -> None:
    gt_path = ROOT / "Данные" / video / f"{video}.csv"
    pred_path = ROOT / "ml" / "output" / f"final_eval_{video}.csv"
    ocr_path = ROOT / "ml" / "output" / f"ocr_qr_{video}.csv"
    qr_path = ROOT / "ml" / "output" / "tracks" / f"track_summary_qr_{video}.csv"
    video_file = find_video(f"{video}.mp4")

    for p, label in ((gt_path, "GT"), (pred_path, "final_eval"),
                     (ocr_path, "ocr_qr"), (qr_path, "track_summary_qr")):
        if not p.is_file():
            raise SystemExit(f"Нет файла {label}: {p}")
    if video_file is None:
        raise SystemExit(f"Нет видео {video}.mp4 под Данные/")

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    ocr = pd.read_csv(ocr_path)
    qr = pd.read_csv(qr_path)

    pairs = match_pairs(gt, pred)
    cap = cv2.VideoCapture(str(video_file))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"=== barcode_recovery_probe v2: {video} ===")
    print(f"video: {video_file.name}  zxing: {'да' if HAS_ZXING else 'нет'}")
    print(f"GT с barcode: {gt['barcode'].astype(str).str.len().gt(5).sum()}")
    print(f"matched pred↔GT пар: {len(pairs)}  (probe first {limit})\n")

    stats = {"crop": Counter(), "video": Counter(), "either": Counter()}
    t0 = time.time()
    n = 0

    for pi, gi, mtype in pairs:
        if n >= limit:
            break
        g = gt.loc[gi]
        p = pred.loc[pi]
        gtbc = re.sub(r"\D", "", str(g.get("barcode", "")))
        if len(gtbc) < 12:
            continue

        hits_crop: Counter[str] = Counter()
        hits_video: Counter[str] = Counter()

        tid = _ocr_track_for_pred(ocr, p)
        crop_ok = 0
        if tid is not None:
            rows = qr[qr["track_id"] == tid].head(ranks)
            for _, r in rows.iterrows():
                for col in ("qr_crop", "hires_crop"):
                    path = _resolve(str(r.get(col, "")))
                    if path is None:
                        continue
                    img = cv2.imread(str(path))
                    if img is None:
                        continue
                    crop_ok += 1
                    for code in decode_image(img, fast=fast):
                        hits_crop[code] += 1

        ts = float(g["frame_timestamp"]) if _to_float(g["frame_timestamp"]) else float(p["frame_timestamp"])
        vimg = _video_crop(cap, ts, _gt_box(g), W, H)
        if vimg is not None:
            for code in decode_image(vimg, fast=fast):
                hits_video[code] += 1

        sc = _status(gtbc, hits_crop)
        sv = _status(gtbc, hits_video)
        combined = hits_crop + hits_video
        se = _status(gtbc, combined)

        stats["crop"][sc.split("(")[0]] += 1
        stats["video"][sv.split("(")[0]] += 1
        stats["either"][se.split("(")[0]] += 1

        name = str(g.get("product_name", ""))[:36]
        print(f"  [{mtype}] track={tid} crops={crop_ok}  gt={gtbc}")
        print(f"         crop → {sc} {dict(hits_crop) if hits_crop else '{}'}")
        print(f"         video→ {sv} {dict(hits_video) if hits_video else '{}'}")
        print(f"         {name}")
        n += 1

    cap.release()
    elapsed = time.time() - t0

    def _exact(c):
        return c.get("EXACT", 0) + c.get("PREFIX12", 0)

    print(f"\n--- итог за {elapsed:.0f}s, проверено {n} GT-пар ---")
    print(f"  crop:   exact/prefix {_exact(stats['crop'])}/{n}")
    print(f"  video:  exact/prefix {_exact(stats['video'])}/{n}")
    print(f"  either: exact/prefix {_exact(stats['either'])}/{n}")

    if _exact(stats["either"]) >= max(2, n // 3):
        print("\n→ ИМЕЕТ СМЫСЛ: decode иногда работает — стоит full barcode_recovery.")
    elif stats["either"].get("ANY", 0) + sum(v for k, v in stats["either"].items() if k.startswith("ANY")) >= 1:
        print("\n→ Слабый сигнал: что-то декодируется, но не GT — нужен digit-OCR / другой crop.")
    else:
        print("\n→ Decode пустой даже с video GT-crop — вариант 1 без обучения CNN, скорее всего, не окупится.")
        print("  Лучше: multi-frame fusion + postprocess на 26_12-20 (уже 25/157).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="25_12-20")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--fast", action="store_true", help="меньше поворотов/масштабов")
    args = ap.parse_args()
    probe(args.video, limit=args.limit, ranks=args.ranks, fast=args.fast)


if __name__ == "__main__":
    main()
