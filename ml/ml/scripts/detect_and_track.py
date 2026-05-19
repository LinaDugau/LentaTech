"""День 2: инференс YOLO + ByteTrack по видео + выбор лучшего кадра."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "ml" / "output" / "runs" / "pricetag_v2" / "weights" / "best.pt"


def sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def process(video_path: Path, model: YOLO, conf: float, imgsz: int,
            out_dir: Path, max_long_side: int,
            min_track_len: int = 2, final_iou: float = 0.4,
            top_k: int = 3):
    """top_k: сколько лучших кадров (по area×sharpness) хранить на track_id.
    Это позволяет потом OCR'ить несколько кадров и majority-voting'ом
    устранить ошибки одного кадра.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    top: dict[int, list[dict]] = {}
    track_len: dict[int, int] = {}

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        H0, W0 = frame.shape[:2]
        scale = max_long_side / max(H0, W0)
        if scale < 1.0:
            frame_s = cv2.resize(frame, (int(W0 * scale), int(H0 * scale)),
                                 interpolation=cv2.INTER_AREA)
        else:
            frame_s = frame
            scale = 1.0

        res = model.track(frame_s, conf=conf, iou=0.5, imgsz=imgsz,
                          persist=True, tracker="bytetrack.yaml",
                          verbose=False)[0]
        if res.boxes is None or res.boxes.id is None:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int)
        ts_ms = int(round(frame_idx / fps * 1000))

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2 = min(frame_s.shape[1] - 1, int(x2))
            y2 = min(frame_s.shape[0] - 1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            track_len[tid] = track_len.get(tid, 0) + 1
            crop = frame_s[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)
            sh = sharpness(crop)
            score = area * sh
            rec = {
                "score": score,
                "crop": crop.copy(),
                "bbox_scaled": (x1, y1, x2, y2),
                "ts_ms": ts_ms,
                "frame_idx": frame_idx,
                "scale": scale,
                "orig_w": W0,
                "orig_h": H0,
                "sharpness": sh,
                "area": area,
            }
            lst = top.setdefault(int(tid), [])
            lst.append(rec)
            if len(lst) > top_k:
                lst.sort(key=lambda r: -r["score"])
                top[int(tid)] = lst[:top_k]

    cap.release()
    for tid in top:
        top[tid].sort(key=lambda r: -r["score"])
    best = {tid: lst[0] for tid, lst in top.items()}
    raw_n = len(best)

    best = {tid: b for tid, b in best.items() if track_len.get(tid, 0) >= min_track_len}
    top = {tid: lst for tid, lst in top.items() if tid in best}

    items = sorted(best.items(), key=lambda kv: -kv[1]["score"])
    kept_ids: list[int] = []
    kept: list[tuple[int, dict]] = []
    for tid, b in items:
        x1, y1, x2, y2 = b["bbox_scaled"]
        keep = True
        for _, k in kept:
            kx1, ky1, kx2, ky2 = k["bbox_scaled"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            ua = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if ua > 0 and inter / ua >= final_iou:
                keep = False
                break
        if keep:
            kept.append((tid, b))
            kept_ids.append(tid)
    best = dict(kept)
    top = {tid: top[tid] for tid in kept_ids}
    print(f"  tracks: raw={raw_n}  after_len>={min_track_len}_and_nms={len(best)}  top_k={top_k}")

    crop_dir = out_dir / "best_frames" / video_path.stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"track_summary_{video_path.stem}.csv"
    fields = ["video", "track_id", "rank", "frame_idx", "frame_ts_ms",
              "x_min", "y_min", "x_max", "y_max",
              "x_min_orig", "y_min_orig", "x_max_orig", "y_max_orig",
              "sharpness", "area", "crop_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tid, lst in top.items():
            for rank, b in enumerate(lst):
                crop_path = crop_dir / f"track_{tid:04d}_r{rank}.jpg"
                cv2.imwrite(str(crop_path), b["crop"],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                x1, y1, x2, y2 = b["bbox_scaled"]
                s = b["scale"]
                w.writerow({
                    "video": video_path.name,
                    "track_id": int(tid),
                    "rank": rank,
                    "frame_idx": b["frame_idx"],
                    "frame_ts_ms": b["ts_ms"],
                    "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                    "x_min_orig": int(x1 / s), "y_min_orig": int(y1 / s),
                    "x_max_orig": int(x2 / s), "y_max_orig": int(y2 / s),
                    "sharpness": round(b["sharpness"], 1),
                    "area": int(b["area"]),
                    "crop_path": str(crop_path.relative_to(ROOT)),
                })
    print(f"  saved -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max_long_side", type=int, default=1920)
    args = ap.parse_args()

    out_dir = ROOT / "ml" / "output" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights: {args.weights}")
    model = YOLO(args.weights)

    if args.videos:
        videos = [Path(v).resolve() for v in args.videos]
    else:
        videos = sorted((ROOT / "Данные").rglob("*.mp4"))

    for v in videos:
        print(f"\n=== {v.name} ===")
        process(v, model, args.conf, args.imgsz, out_dir, args.max_long_side)


if __name__ == "__main__":
    main()
