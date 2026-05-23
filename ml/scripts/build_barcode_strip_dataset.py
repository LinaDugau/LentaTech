"""Сбор датасета digit-strip из GT bbox + crop'ов треков."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import extract_digit_strip, normalize_barcode  # noqa: E402
from video_resolve import find_video  # noqa: E402

DATA = ROOT / "Данные"
OUT_DIR = ROOT / "ml" / "data" / "barcode_strips"
VIDEOS = ("25_12-20", "26_12-20", "43_15")


def _read_gt(video: str) -> pd.DataFrame:
    p = DATA / video / f"{video}.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"barcode": str}, keep_default_na=False)


def _bbox_from_row(r) -> tuple[int, int, int, int]:
    def f(name):
        return int(float(str(getattr(r, name, 0)).replace(",", ".")))

    return f("x_min"), f("y_min"), f("x_max"), f("y_max")


def _crop_from_video(cap: cv2.VideoCapture, ts_ms: int, bbox, expand_down: float = 0.18):
    x0, y0, x1, y1 = bbox
    h_tag = max(1, y1 - y0)
    y1 = int(y1 + h_tag * expand_down)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, ts_ms))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fw, x1), min(fh, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def build_from_gt() -> list[dict]:
    rows: list[dict] = []
    for video in VIDEOS:
        gt = _read_gt(video)
        if gt.empty:
            print(f"  skip GT {video}: no file")
            continue
        vpath = find_video(f"{video}.mp4") or find_video(f"{video}/2.mp4")
        if vpath is None:
            for pat in (f"{video}.mp4", "2.mp4", "1.mp4"):
                vpath = find_video(f"{video}/{pat}") if "/" not in pat else find_video(pat)
                if vpath:
                    break
        if vpath is None:
            sample = str(gt.iloc[0].get("filename", ""))
            vname = Path(sample).name if sample else f"{video}.mp4"
            vpath = find_video(vname) or find_video(f"{video}/{vname}")
        if vpath is None:
            print(f"  skip GT {video}: video not found")
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"  skip GT {video}: cannot open {vpath}")
            continue
        n_ok = 0
        for i, r in enumerate(gt.itertuples(index=False)):
            bc = normalize_barcode(getattr(r, "barcode", ""))
            if len(bc) < 12:
                continue
            if len(bc) == 12:
                bc = bc  # train as 12-digit strip (pad visually in model)
            ts = int(float(str(getattr(r, "frame_timestamp", 0)).replace(",", ".")))
            crop = _crop_from_video(cap, ts, _bbox_from_row(r))
            if crop is None:
                continue
            strip = extract_digit_strip(crop)
            if strip is None:
                continue
            sid = f"gt_{video}_{i:04d}"
            rows.append({
                "id": sid,
                "source": "gt_video",
                "video": video,
                "barcode": bc[:13],
                "strip_path": f"strips/{sid}.png",
            })
            n_ok += 1
        cap.release()
        print(f"  GT {video}: {n_ok} strips from {vpath.name}")
    return rows


def build_from_pred_crops() -> list[dict]:
    rows: list[dict] = []
    for video in VIDEOS:
        ocr_path = ROOT / "ml" / "output" / f"ocr_qr_{video}.csv"
        if not ocr_path.is_file():
            continue
        ocr = pd.read_csv(ocr_path)
        if "barcode_raw" not in ocr.columns:
            continue
        for i, r in enumerate(ocr.itertuples(index=False)):
            bc = normalize_barcode(getattr(r, "barcode_raw", ""))
            if len(bc) < 12:
                continue
            cp = getattr(r, "hires_crop", None) or getattr(r, "crop_path", None)
            if not isinstance(cp, str) or not cp.strip():
                continue
            img = cv2.imread(str(ROOT / cp))
            strip = extract_digit_strip(img)
            if strip is None:
                continue
            sid = f"pred_{video}_{int(r.track_id):04d}_r{i % 1000:03d}"
            rows.append({
                "id": sid,
                "source": "pred_crop",
                "video": video,
                "barcode": bc[:13],
                "strip_path": f"strips/{sid}.png",
            })
    print(f"  pred crops: {len(rows)} strips")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-pred-crops", action="store_true")
    ap.add_argument("--ocr-dir", type=Path, default=ROOT / "ml" / "output",
                    help="ocr_qr_*.csv source (e.g. _heavy_all3)")
    ap.add_argument("--all-decode-frames", action="store_true",
                    help="keep every frame with barcode_raw, not one per track")
    ap.add_argument("--merge-pretrain", type=Path, default=None,
                    help="merge labels from catalog pretrain dir")
    ap.add_argument("--merge-external", type=Path, default=None,
                    help="merge labels from external import dir")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strips").mkdir(parents=True, exist_ok=True)

    final_rows: list[dict] = []

    for video in VIDEOS:
        gt = _read_gt(video)
        if gt.empty:
            continue
        vpath = None
        for cand in (f"{video}.mp4", "2.mp4"):
            vpath = find_video(cand) or find_video(f"{video}/{cand}")
            if vpath:
                break
        if vpath is None:
            sample = str(gt.iloc[0].get("filename", ""))
            vpath = find_video(Path(sample).name) if sample else None
        if vpath is None:
            print(f"  skip GT {video}: video not found")
            continue
        cap = cv2.VideoCapture(str(vpath))
        n_ok = 0
        for i, r in enumerate(gt.itertuples(index=False)):
            bc = normalize_barcode(getattr(r, "barcode", ""))
            if len(bc) < 12:
                continue
            ts = int(float(str(getattr(r, "frame_timestamp", 0)).replace(",", ".")))
            crop = _crop_from_video(cap, ts, _bbox_from_row(r)) if cap.isOpened() else None
            if crop is None:
                continue
            strip = extract_digit_strip(crop)
            if strip is None:
                continue
            sid = f"gt_{video}_{i:04d}"
            rel = f"strips/{sid}.png"
            cv2.imwrite(str(args.out / rel), strip)
            final_rows.append({
                "id": sid, "source": "gt_video", "video": video,
                "barcode": bc[:13], "strip_path": rel,
            })
            n_ok += 1
        cap.release()
        print(f"  GT {video}: {n_ok} strips from {vpath.name}")

    if args.include_pred_crops:
        for video in VIDEOS:
            ocr_path = args.ocr_dir / f"ocr_qr_{video}.csv"
            if not ocr_path.is_file():
                print(f"  skip pred {video}: no {ocr_path}")
                continue
            ocr = pd.read_csv(ocr_path)
            seen: set[tuple] = set()
            n_pred = 0
            for i, r in enumerate(ocr.itertuples(index=False)):
                bc = normalize_barcode(getattr(r, "barcode_raw", ""))
                if len(bc) < 12:
                    continue
                key = (video, int(r.track_id), bc)
                if not args.all_decode_frames and key in seen:
                    continue
                seen.add(key)
                cp = getattr(r, "hires_crop", None) or getattr(r, "crop_path", None)
                if not isinstance(cp, str):
                    continue
                img = cv2.imread(str(ROOT / cp))
                strip = extract_digit_strip(img)
                if strip is None:
                    continue
                sid = f"pred_{video}_{int(r.track_id):04d}"
                if args.all_decode_frames:
                    sid = f"{sid}_f{i:04d}"
                rel = f"strips/{sid}.png"
                cv2.imwrite(str(args.out / rel), strip)
                final_rows.append({
                    "id": sid, "source": "pred_crop", "video": video,
                    "barcode": bc[:13], "strip_path": rel,
                })
                n_pred += 1
            print(f"  pred {video}: {n_pred} strips from {ocr_path.parent.name}")

    def _merge_labels(extra_dir: Path, prefix: str):
        lp = extra_dir / "labels.csv"
        if not lp.is_file():
            print(f"  skip merge {prefix}: no {lp}")
            return
        extra = pd.read_csv(lp)
        n = 0
        for _, row in extra.iterrows():
            src = extra_dir / str(row["strip_path"])
            if not src.is_file():
                continue
            sid = f"{prefix}_{row['id']}"
            rel = f"strips/{sid}.png"
            dst = args.out / rel
            if not dst.is_file():
                import shutil
                shutil.copy2(src, dst)
            final_rows.append({
                "id": sid,
                "source": str(row.get("source", prefix)),
                "video": str(row.get("video", "")),
                "barcode": str(row["barcode"])[:13],
                "strip_path": rel,
            })
            n += 1
        print(f"  merged {prefix}: {n} strips from {extra_dir.name}")

    if args.merge_pretrain:
        _merge_labels(args.merge_pretrain, "pt")
    if args.merge_external:
        _merge_labels(args.merge_external, "ext")

    labels_path = args.out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "video", "barcode", "strip_path"])
        w.writeheader()
        w.writerows(final_rows)

    print(f"Saved {len(final_rows)} strips → {args.out}")


if __name__ == "__main__":
    main()
