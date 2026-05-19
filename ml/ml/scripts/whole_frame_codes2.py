"""Whole-frame QR/barcode decode на 4K кадре (WeChat + zxing + cv2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from video_resolve import find_video  # noqa: E402
from build_final_csv import choose_best_barcode, choose_best_qr  # noqa: E402

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

_WECHAT = None
_QD = cv2.QRCodeDetector()
_BD = cv2.barcode.BarcodeDetector()


def _get_wechat():
    global _WECHAT
    if _WECHAT is None:
        try:
            _WECHAT = cv2.wechat_qrcode_WeChatQRCode(
                str(WEIGHTS / "detect.prototxt"),
                str(WEIGHTS / "detect.caffemodel"),
                str(WEIGHTS / "sr.prototxt"),
                str(WEIGHTS / "sr.caffemodel"),
            )
        except Exception as e:
            print(f"  WeChat load failed: {e}")
            _WECHAT = False
    return _WECHAT if _WECHAT is not False else None


def in_or_near(track_box, code_box, margin: float = 1.0) -> bool:
    """QR чаще над ценником — сильно расширяем bbox вверх."""
    tx1, ty1, tx2, ty2 = track_box
    cx1, cy1, cx2, cy2 = code_box
    tw, th = tx2 - tx1, ty2 - ty1
    ix1, iy1 = max(tx1, cx1), max(ty1, cy1)
    ix2, iy2 = min(tx2, cx2), min(ty2, cy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    code_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
    if inter / code_area >= 0.35:
        return True
    ext = (
        tx1 - tw * margin,
        ty1 - th * (margin + 0.6),
        tx2 + tw * margin,
        ty2 + th * 0.4,
    )
    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    return ext[0] <= cx <= ext[2] and ext[1] <= cy <= ext[3]


def _rotate(frame, rot: int):
    if rot == 0:
        return frame
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _to_orig(box, rot: int, w: int, h: int):
    x1, y1, x2, y2 = box
    if rot == 0:
        return (x1, y1, x2, y2)
    if rot == 90:
        return (y1, w - x2, y2, w - x1)
    if rot == 180:
        return (w - x2, h - y2, w - x1, h - y1)
    return (h - y2, x1, h - y1, x2)


def decode_frame(frame, w: int, h: int) -> tuple[list[tuple[str, tuple]], list[tuple[str, tuple]]]:
    codes_b: list[tuple[str, tuple[float, float, float, float]]] = []
    codes_q: list[tuple[str, tuple[float, float, float, float]]] = []
    seen_q: set[str] = set()
    seen_b: set[str] = set()

    wechat = _get_wechat()
    if wechat is not None:
        try:
            results, points = wechat.detectAndDecode(frame)
            for s, pts in zip(results, points):
                if not s or s in seen_q:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                codes_q.append((s, (min(xs), min(ys), max(xs), max(ys))))
                seen_q.add(s)
        except Exception:
            pass

    for rot in (0, 90, 180, 270):
        fr = _rotate(frame, rot)
        if HAS_ZXING:
            try:
                for r in _zxing.read_barcodes(fr):
                    if not r.text:
                        continue
                    s = r.text.strip()
                    fmt = str(r.format).lower()
                    pos = r.position
                    xs = [p.x for p in pos]
                    ys = [p.y for p in pos]
                    bb = _to_orig((min(xs), min(ys), max(xs), max(ys)), rot, w, h)
                    if "qr" in fmt:
                        if s not in seen_q:
                            codes_q.append((s, bb))
                            seen_q.add(s)
                    elif s not in seen_b:
                        codes_b.append((s, bb))
                        seen_b.add(s)
            except Exception:
                pass

        if HAS_PYZBAR:
            try:
                for r in _zbar.decode(fr):
                    if not r.data:
                        continue
                    s = r.data.decode("utf-8", errors="ignore").strip()
                    bb = _to_orig(
                        (r.rect.left, r.rect.top,
                         r.rect.left + r.rect.width, r.rect.top + r.rect.height),
                        rot, w, h,
                    )
                    if r.type == "QRCODE":
                        if s not in seen_q:
                            codes_q.append((s, bb))
                            seen_q.add(s)
                    elif s not in seen_b:
                        codes_b.append((s, bb))
                        seen_b.add(s)
            except Exception:
                pass

        try:
            ok_b, decoded_b, _, corners_b = _BD.detectAndDecodeWithType(fr)
            if ok_b and decoded_b is not None and corners_b is not None:
                for s, c in zip(decoded_b, corners_b):
                    if not s or s in seen_b:
                        continue
                    xs, ys = c[:, 0], c[:, 1]
                    bb = _to_orig((xs.min(), ys.min(), xs.max(), ys.max()), rot, w, h)
                    codes_b.append((s, bb))
                    seen_b.add(s)
        except Exception:
            pass

        try:
            ok_q, infos, corners_q, _ = _QD.detectAndDecodeMulti(fr)
            if ok_q and infos is not None and corners_q is not None:
                for s, c in zip(infos, corners_q):
                    if not s or s in seen_q:
                        continue
                    xs, ys = c[:, 0], c[:, 1]
                    bb = _to_orig((xs.min(), ys.min(), xs.max(), ys.max()), rot, w, h)
                    codes_q.append((s, bb))
                    seen_q.add(s)
        except Exception:
            pass

    return codes_b, codes_q


def _propagate_tracks(df: pd.DataFrame) -> None:
    """Заполнить пустые qr/barcode на всех K кадрах трека лучшим значением."""
    if "track_id" not in df.columns:
        return
    for tid, sub in df.groupby("track_id"):
        qrs = [str(x) for x in sub["qr_raw"] if str(x).strip() and str(x) != "nan"]
        bcs = [str(x) for x in sub["barcode_raw"] if str(x).strip() and str(x) != "nan"]
        best_q = choose_best_qr(qrs)
        best_b = choose_best_barcode(bcs)
        mask = df["track_id"] == tid
        if best_q:
            empty_q = df.loc[mask, "qr_raw"].astype(str).str.strip().isin(("", "nan"))
            df.loc[mask & empty_q, "qr_raw"] = best_q
        if best_b:
            empty_b = df.loc[mask, "barcode_raw"].astype(str).str.strip().isin(("", "nan"))
            df.loc[mask & empty_b, "barcode_raw"] = best_b


def process(ocr_csv: Path) -> tuple[int, int]:
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return 0, 0
    for c in ("barcode_raw", "qr_raw"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
        else:
            df[c] = ""

    video_name = str(df.iloc[0]["video"])
    stem = ocr_csv.stem.replace("ocr_qr_", "")
    video = find_video(video_name)
    if video is None:
        print(f"  video not found: {video_name}")
        return 0, 0

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df["frame_idx_compute"] = (df["frame_ts_ms"] * fps / 1000).round().astype(int)
    unique_frames = sorted(df["frame_idx_compute"].unique())

    n_bc_new = 0
    n_qr_new = 0
    n_frames_with_codes = 0

    for fi in tqdm(unique_frames, desc=stem):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        codes_b, codes_q = decode_frame(frame, w, h)
        if not codes_b and not codes_q:
            continue
        n_frames_with_codes += 1

        rows_for_frame = df[df["frame_idx_compute"] == fi]
        for idx, r in rows_for_frame.iterrows():
            tb = (float(r.x_min_orig), float(r.y_min_orig),
                  float(r.x_max_orig), float(r.y_max_orig))
            if not str(df.at[idx, "barcode_raw"]).strip():
                for s, cb in codes_b:
                    if in_or_near(tb, cb):
                        df.at[idx, "barcode_raw"] = s
                        n_bc_new += 1
                        break
            if not str(df.at[idx, "qr_raw"]).strip():
                for s, cb in codes_q:
                    if in_or_near(tb, cb):
                        df.at[idx, "qr_raw"] = s
                        n_qr_new += 1
                        break

    cap.release()
    _propagate_tracks(df)
    df = df.drop(columns=["frame_idx_compute"], errors="ignore")
    df.to_csv(ocr_csv, index=False)

    qr_total = (df["qr_raw"].astype(str).str.len() > 5).sum()
    print(f"  frames_with_codes={n_frames_with_codes}/{len(unique_frames)}")
    print(f"  decoded new: barcodes={n_bc_new}, QRs={n_qr_new}, qr_total={qr_total}/{len(df)}")
    return n_bc_new, n_qr_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--video", action="append")
    args = ap.parse_args()
    if args.files:
        files = [Path(p) for p in args.files]
    elif args.video:
        files = [ROOT / "ml" / "output" / f"ocr_qr_{v}.csv" for v in args.video]
    else:
        files = sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv"))
        files = [f for f in files if "_ml" not in f.stem and f.stem != "ocr_qr_input"]

    total_bc = total_qr = 0
    for f in files:
        print(f"\n=== {f.name} ===")
        bc, qr = process(f)
        total_bc += bc
        total_qr += qr
    print(f"\nTotal new: barcodes={total_bc}, QRs={total_qr}")


if __name__ == "__main__":
    main()
