"""Извлечение background-кадров из тестовых видео для синтеза."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default=str(ROOT / "Данные"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "output" / "synth" / "backgrounds"))
    ap.add_argument("--step", type=int, default=25,
                    help="каждый N-й кадр (для разнообразия)")
    ap.add_argument("--max_per_video", type=int, default=80)
    ap.add_argument("--max_long_side", type=int, default=1920)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(7)

    total = 0
    for video_path in sorted(Path(args.videos_dir).rglob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = args.max_long_side / max(W0, H0)
        Wn, Hn = int(W0 * scale), int(H0 * scale)
        per_video = 0
        for fi in tqdm(range(0, n_frames, args.step),
                       desc=video_path.parent.name + "/" + video_path.name,
                       leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (Wn, Hn),
                                    interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{video_path.parent.name}__{video_path.stem}_{fi:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            total += 1
            per_video += 1
            if per_video >= args.max_per_video:
                break
        cap.release()
    print(f"\n✅ Backgrounds: {total} → {out_dir}")


if __name__ == "__main__":
    main()
