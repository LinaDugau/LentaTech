"""Извлечение кадров."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
OUT = ROOT / "ml" / "output" / "frames"
INDEX = ROOT / "ml" / "output" / "frames_index.csv"


def sharpness(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract(video_path: Path, stride_s: float, sharp_min: float,
            diff_min: float, max_long_side: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(stride_s * fps)))

    out_dir = OUT / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    prev_small: np.ndarray | None = None
    pbar = tqdm(total=n_frames, desc=video_path.name, unit="f")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        if idx % stride != 0:
            idx += 1
            continue

        h, w = frame.shape[:2]
        scale = max_long_side / max(h, w)
        if scale < 1.0:
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame
            scale = 1.0

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        sh = sharpness(gray)
        if sh < sharp_min:
            idx += 1
            continue

        small = cv2.resize(gray, (160, 90))
        if prev_small is not None:
            diff = float(np.mean(cv2.absdiff(small, prev_small)))
            if diff < diff_min:
                idx += 1
                continue
        prev_small = small

        ts_ms = int(round(idx / fps * 1000))
        fname = f"{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame_small,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({
            "video": video_path.relative_to(ROOT).as_posix(),
            "frame_idx": idx,
            "frame_ts_ms": ts_ms,
            "saved_path": (out_dir / fname).relative_to(ROOT).as_posix(),
            "orig_w": w, "orig_h": h,
            "scale": round(scale, 4),
            "sharpness": round(sh, 1),
        })
        idx += 1
    pbar.close()
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride_s", type=float, default=0.25,
                    help="Сэмплинг: каждые N секунд")
    ap.add_argument("--sharp_min", type=float, default=40.0,
                    help="Минимальная резкость (variance of Laplacian)")
    ap.add_argument("--diff_min", type=float, default=3.0,
                    help="Минимальная разница с предыдущим сохранённым кадром")
    ap.add_argument("--max_long_side", type=int, default=1920,
                    help="Уменьшение длинной стороны (исходник 3840)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Конкретные .mp4; иначе все из Данные/")
    args = ap.parse_args()

    if args.videos:
        videos = [Path(v).resolve() for v in args.videos]
    else:
        videos = sorted(DATA.rglob("*.mp4"))

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for v in videos:
        all_rows.extend(extract(v, args.stride_s, args.sharp_min,
                                args.diff_min, args.max_long_side))

    if all_rows:
        with INDEX.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} frames. Index: {INDEX}")


if __name__ == "__main__":
    main()
