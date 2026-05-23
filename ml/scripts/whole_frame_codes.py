"""Whole-frame barcode/QR."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
from ml.scripts.video_resolve import find_video  # noqa: E402


def in_or_near(track_box, code_box):
    tx1, ty1, tx2, ty2 = track_box
    cx1, cy1, cx2, cy2 = code_box
    ix1, iy1 = max(tx1, cx1), max(ty1, cy1)
    ix2, iy2 = min(tx2, cx2), min(ty2, cy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    code_area = (cx2 - cx1) * (cy2 - cy1)
    if code_area > 0 and inter / code_area >= 0.5:
        return True
    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    margin_x = (tx2 - tx1) * 0.25
    margin_y = (ty2 - ty1) * 0.25
    return (tx1 - margin_x <= cx <= tx2 + margin_x and
            ty1 - margin_y <= cy <= ty2 + margin_y)


def process(ocr_csv: Path):
    df = pd.read_csv(ocr_csv)
    if df.empty:
        return
    video_name = df.iloc[0]["video"]
    stem = ocr_csv.stem.replace("ocr_qr_", "")
    video = find_video(video_name)
    if video is None:
        print(f"  no video for {video_name}")
        return

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    bd = cv2.barcode.BarcodeDetector()
    qd = cv2.QRCodeDetector()

    df["frame_idx_compute"] = (df["frame_ts_ms"] * fps / 1000).round().astype(int)
    unique_frames = sorted(df["frame_idx_compute"].unique())

    bc_by_track: dict[int, str] = {}
    qr_by_track: dict[int, str] = {}

    H_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def to_orig(box, rot, scale):
        x1, y1, x2, y2 = box
        x1, x2 = x1 / scale, x2 / scale
        y1, y2 = y1 / scale, y2 / scale
        if rot == 0:
            return (x1, y1, x2, y2)
        if rot == 90:  # cw: (x,y)_rot = (H - y, x)
            return (y1, W_frame - x2, y2, W_frame - x1)
        if rot == 180:
            return (W_frame - x2, H_frame - y2, W_frame - x1, H_frame - y1)
        if rot == 270:
            return (H_frame - y2, x1, H_frame - y1, x2)
        return (x1, y1, x2, y2)

    for fi in tqdm(unique_frames, desc=stem):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue

        codes_b: list[tuple[str, tuple[float, float, float, float]]] = []
        codes_q: list[tuple[str, tuple[float, float, float, float]]] = []
        for rot in (0, 90, 180, 270):
            if rot == 0:
                fr = frame
            elif rot == 90:
                fr = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                fr = cv2.rotate(frame, cv2.ROTATE_180)
            else:
                fr = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            ok_b, decoded_b, _, corners_b = bd.detectAndDecodeWithType(fr)
            if ok_b and decoded_b is not None and corners_b is not None:
                for s, c in zip(decoded_b, corners_b):
                    if not s:
                        continue
                    xs = c[:, 0]; ys = c[:, 1]
                    bb = (xs.min(), ys.min(), xs.max(), ys.max())
                    codes_b.append((s, to_orig(bb, rot, 1.0)))

            ok_q, infos, corners_q, _ = qd.detectAndDecodeMulti(fr)
            if ok_q and infos is not None and corners_q is not None:
                for s, c in zip(infos, corners_q):
                    if not s:
                        continue
                    xs = c[:, 0]; ys = c[:, 1]
                    bb = (xs.min(), ys.min(), xs.max(), ys.max())
                    codes_q.append((s, to_orig(bb, rot, 1.0)))

        if not codes_b and not codes_q:
            continue

        rows = df[df["frame_idx_compute"] == fi]
        for _, r in rows.iterrows():
            tb = (r.x_min_orig, r.y_min_orig, r.x_max_orig, r.y_max_orig)
            for s, cb in codes_b:
                if in_or_near(tb, cb) and r.track_id not in bc_by_track:
                    bc_by_track[r.track_id] = s
            for s, cb in codes_q:
                if in_or_near(tb, cb) and r.track_id not in qr_by_track:
                    qr_by_track[r.track_id] = s
    cap.release()

    def coalesce(old, new):
        if isinstance(old, str) and old.strip():
            return old
        return new

    df["barcode_raw"] = [coalesce(r.barcode_raw, bc_by_track.get(r.track_id, ""))
                        for r in df.itertuples()]
    df["qr_raw"] = [coalesce(r.qr_raw, qr_by_track.get(r.track_id, ""))
                    for r in df.itertuples()]
    df = df.drop(columns=["frame_idx_compute"])
    df.to_csv(ocr_csv, index=False)
    n_b = sum(1 for v in df["barcode_raw"] if isinstance(v, str) and v.strip())
    n_q = sum(1 for v in df["qr_raw"] if isinstance(v, str) and v.strip())
    print(f"  barcodes: {n_b}/{len(df)}, QRs: {n_q}/{len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()
    files = [Path(p) for p in args.files] if args.files else \
        sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv"))
    for f in files:
        print(f"\n=== {f.name} ===")
        process(f)


if __name__ == "__main__":
    main()
