"""День 13: YOLO inference в режиме tiling 2×2 + основной кадр."""
from __future__ import annotations

import argparse
import csv
import sys
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


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.maximum(0, x2 - x1); ih = np.maximum(0, y2 - y1)
        inter = iw * ih
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ua = a + b - inter
        iou = np.where(ua > 0, inter / ua, 0)
        idxs = rest[iou < iou_thr]
    return keep


def detect_tiled(model: YOLO, frame: np.ndarray, conf: float, imgsz: int,
                 tiles: int = 2, overlap: float = 0.2,
                 tta: bool = False, models: list = None):
    """Возвращает (boxes_xyxy_orig, scores).

    tta=True: каждый tile прогоняется на 3 вариантах (исходный + ×0.85 + ×1.20 scale)
    + горизонтальное отражение. Это +5-10% recall ценой ~3× времени.

    models=[m1, m2, ...]: если задан, прогоняем КАЖДЫЙ tile через все модели
    и объединяем bbox (ensemble). Например v2 + v4.
    """
    H, W = frame.shape[:2]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    model_list = models if models else [model]

    tw = int(W / tiles * (1 + overlap))
    th = int(H / tiles * (1 + overlap))
    step_x = (W - tw) // (tiles - 1) if tiles > 1 else 0
    step_y = (H - th) // (tiles - 1) if tiles > 1 else 0

    def _run(img, m):
        res = m.predict(img, conf=conf, iou=0.5, imgsz=imgsz,
                        augment=tta, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()

    for ty in range(tiles):
        for tx in range(tiles):
            x0 = tx * step_x
            y0 = ty * step_y
            x1 = min(x0 + tw, W)
            y1 = min(y0 + th, H)
            tile = frame[y0:y1, x0:x1]
            for m in model_list:
                xyxy, sc = _run(tile, m)
                if xyxy is None:
                    continue
                xyxy = xyxy.copy()
                xyxy[:, [0, 2]] += x0
                xyxy[:, [1, 3]] += y0
                all_boxes.append(xyxy)
                all_scores.append(sc)

    scale = min(imgsz / W, imgsz / H, 1.0)
    if scale < 1.0:
        small = cv2.resize(frame, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame
    for m in model_list:
        xyxy, sc = _run(small, m)
        if xyxy is None:
            continue
        xyxy = xyxy.copy()
        if scale < 1.0:
            xyxy /= scale
        all_boxes.append(xyxy)
        all_scores.append(sc)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0)
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(boxes, scores, iou_thr=0.45)
    return boxes[keep], scores[keep]


def simple_track(detections_per_frame: list[list[tuple[np.ndarray, float, int]]],
                  iou_match: float = 0.30, max_skip: int = 5):
    """Простой жадный трекер по IoU между соседними кадрами.

    detections_per_frame: список по кадрам [(bbox, score, frame_idx), ...]
    Возвращает: dict track_id -> [(bbox, score, frame_idx), ...]
    """
    tracks: dict[int, list] = {}
    last_box: dict[int, np.ndarray] = {}
    last_frame: dict[int, int] = {}
    next_id = 0
    for frame_dets in detections_per_frame:
        used = set()
        for bbox, score, fi in frame_dets:
            best_id = None
            best_iou = 0
            for tid, lb in last_box.items():
                if tid in used:
                    continue
                if fi - last_frame[tid] > max_skip:
                    continue
                ix1, iy1 = max(bbox[0], lb[0]), max(bbox[1], lb[1])
                ix2, iy2 = min(bbox[2], lb[2]), min(bbox[3], lb[3])
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                b = (lb[2] - lb[0]) * (lb[3] - lb[1])
                ua = a + b - inter
                v = inter / ua if ua > 0 else 0
                if v > best_iou:
                    best_iou = v
                    best_id = tid
            if best_id is not None and best_iou >= iou_match:
                tid = best_id
            else:
                tid = next_id; next_id += 1
                tracks[tid] = []
            tracks.setdefault(tid, []).append((bbox, score, fi))
            last_box[tid] = bbox
            last_frame[tid] = fi
            used.add(tid)
    return tracks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="основная модель")
    ap.add_argument("--ensemble", nargs="*", default=None,
                    help="доп. модели для ансамбля (.pt).")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation (scale+flip).")
    ap.add_argument("--glare_removal", action="store_true",
                    help="preprocessing: убирает засветы перед YOLO. "
                         "Помогает на ценниках со стеклянными бликами.")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--frame_step", type=int, default=5,
                    help="обрабатываем каждый N-й кадр (для скорости)")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--min_track_len", type=int, default=2)
    args = ap.parse_args()

    out_dir = ROOT / "ml" / "output" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading main: {args.weights}")
    model = YOLO(args.weights)
    models = [model]
    if args.ensemble:
        for w in args.ensemble:
            print(f"Loading ensemble: {w}")
            models.append(YOLO(w))
        print(f"Ensemble of {len(models)} models")
    if args.tta:
        print("TTA enabled (scale + flip)")
    if args.videos:
        videos = [Path(v).resolve() for v in args.videos]
    else:
        videos = sorted((ROOT / "Данные").rglob("*.mp4"))

    for v in videos:
        print(f"\n=== {v.name} ===")
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per_frame: list[list[tuple]] = []
        frame_indices = list(range(0, n_frames, args.frame_step))
        cur = -1
        glare_fn = None
        if args.glare_removal:
            sys.path.insert(0, str(Path(__file__).parent))
            from glare_removal import preprocess_for_detection as glare_fn
            print("  Glare removal: ON")
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            if glare_fn is not None:
                frame = glare_fn(frame)
            boxes, scores = detect_tiled(
                model, frame, args.conf, args.imgsz,
                tta=args.tta, models=models if len(models) > 1 else None)
            dets = [(boxes[i], float(scores[i]), fi) for i in range(len(boxes))]
            per_frame.append(dets)
            print(f"  frame {fi}: {len(dets)} dets", end="\r", flush=True)
        cap.release()
        print()
        tracks = simple_track(per_frame)
        cap = cv2.VideoCapture(str(v))
        kept = []
        for tid, lst in tracks.items():
            if len(lst) < args.min_track_len:
                continue
            scored = []
            for bbox, _, fi in lst:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                sc = (x2 - x1) * (y2 - y1) * sharpness(crop)
                scored.append((sc, fi, (x1, y1, x2, y2), crop))
            if not scored:
                continue
            scored.sort(key=lambda t: -t[0])

            min_gap_frames = max(1, int(round(fps * 0.5)))
            selected = []
            used_idx = set()
            for idx, rec in enumerate(scored):
                if len(selected) >= args.top_k:
                    break
                fi = rec[1]
                if all(abs(fi - prev[1]) >= min_gap_frames for prev in selected):
                    selected.append(rec)
                    used_idx.add(idx)
            for idx, rec in enumerate(scored):
                if len(selected) >= args.top_k:
                    break
                if idx not in used_idx:
                    selected.append(rec)
            kept.append((tid, selected))
        cap.release()

        crop_dir = out_dir / "best_frames" / v.stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"track_summary_{v.stem}.csv"
        fields = ["video", "track_id", "rank", "frame_idx", "frame_ts_ms",
                  "x_min", "y_min", "x_max", "y_max",
                  "x_min_orig", "y_min_orig", "x_max_orig", "y_max_orig",
                  "sharpness", "area", "crop_path"]
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fields)
            w.writeheader()
            for tid, recs in kept:
                for rank, (sc, fi, bb, crop) in enumerate(recs):
                    x1, y1, x2, y2 = bb
                    cp = crop_dir / f"track_{tid:04d}_r{rank}.jpg"
                    cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow({
                        "video": v.name,
                        "track_id": int(tid), "rank": rank,
                        "frame_idx": fi,
                        "frame_ts_ms": int(round(fi / fps * 1000)),
                        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                        "x_min_orig": x1, "y_min_orig": y1,
                        "x_max_orig": x2, "y_max_orig": y2,
                        "sharpness": round(float(sharpness(crop)), 1),
                        "area": int((x2 - x1) * (y2 - y1)),
                        "crop_path": str(cp.relative_to(ROOT)),
                    })
        print(f"  tracks: {len(kept)}")
        print(f"  saved -> {csv_path}")


if __name__ == "__main__":
    main()
