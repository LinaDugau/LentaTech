"""Коррекция дисторсии широкоугольной камеры робота."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SENSOR_DIAGONAL_MM = 16.0 / 2.8     # ≈5.714
FOCAL_MM           = 2.8
DEFAULT_WIDTH      = 3840
DEFAULT_HEIGHT     = 2160
DIST_COEFFS = np.array(
    [-0.276, 0.06, 0.0084, -0.0016, -0.0044], dtype=np.float32
)


def camera_matrix(width: int, height: int,
                  diag_mm: float = SENSOR_DIAGONAL_MM,
                  focal_mm: float = FOCAL_MM) -> np.ndarray:
    """Intrinsic matrix K (pinhole)."""
    ar = width / height
    h_mm = diag_mm / math.sqrt(ar ** 2 + 1)
    w_mm = ar * h_mm
    fx = focal_mm * width / w_mm
    fy = focal_mm * height / h_mm
    return np.array(
        [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )


class UndistortMap:
    """Pre-computed maps для cv2.remap. Считаем один раз, применяем
    бесплатно к каждому кадру (cv2.remap ~50 ms на 4K)."""

    def __init__(self, width: int, height: int, crop_to_roi: bool = False):
        self.K = camera_matrix(width, height)
        self.dist = DIST_COEFFS
        self.size = (width, height)
        self.crop_to_roi = crop_to_roi
        new_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, self.size, alpha=0.0, newImgSize=self.size
        )
        self.new_K = new_K
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, new_K, self.size, cv2.CV_16SC2
        )
        x, y, w, h = self.roi
        print(f"  K  =\n{self.K}")
        print(f"  K' =\n{new_K}")
        print(f"  ROI: x={x}, y={y}, w={w}, h={h}  (potential crop {width-w}px wide, {height-h}px tall)")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        out = cv2.remap(frame, self.map1, self.map2,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT)
        if self.crop_to_roi:
            x, y, w, h = self.roi
            out = out[y:y + h, x:x + w]
        return out


def grab_frame(video: Path, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {frame_idx} from {video}")
    return frame


def side_by_side(orig: np.ndarray, undist: np.ndarray,
                 labels: tuple[str, str] = ("BEFORE", "AFTER")) -> np.ndarray:
    h1, w1 = orig.shape[:2]
    h2, w2 = undist.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        undist = cv2.resize(undist, (int(w2 * scale), h1))
        h2, w2 = undist.shape[:2]
    out = np.zeros((h1, w1 + w2 + 20, 3), dtype=np.uint8)
    out[:, :w1] = orig
    out[:, w1 + 20:w1 + 20 + w2] = undist
    cv2.putText(out, labels[0], (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 255, 0), 5)
    cv2.putText(out, labels[1], (w1 + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 255, 255), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="превью одного кадра до/после на каждом видео")
    ap.add_argument("--video", help="конкретное видео для обработки")
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--out", help="выходной путь (JPG или MP4)")
    ap.add_argument("--to_video", action="store_true",
                    help="undistort всё видео в MP4")
    ap.add_argument("--crop_to_roi", action="store_true",
                    help="обрезать до валидной ROI (без чёрных углов)")
    args = ap.parse_args()

    if args.preview:
        out_dir = ROOT / "ml" / "output" / "undistort_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("25_12-20", "26_12-20", "43_15"):
            v = ROOT / "Данные" / name / f"{name}.mp4"
            if not v.exists():
                print(f"skip: {v} not found")
                continue
            print(f"\n=== {name} (frame {args.frame}) ===")
            frame = grab_frame(v, args.frame)
            h, w = frame.shape[:2]
            ud = UndistortMap(w, h, crop_to_roi=args.crop_to_roi)
            undist = ud(frame)
            cmp = side_by_side(frame, undist)
            scale = 1600 / cmp.shape[1]
            cmp = cv2.resize(cmp, None, fx=scale, fy=scale)
            out_path = out_dir / f"{name}_undistort_compare.jpg"
            cv2.imwrite(str(out_path), cmp, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  saved → {out_path}")
        return

    if not args.video:
        ap.error("--video required (or use --preview)")
    v = Path(args.video)
    frame = grab_frame(v, args.frame)
    h, w = frame.shape[:2]
    ud = UndistortMap(w, h, crop_to_roi=args.crop_to_roi)

    if args.to_video:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_size = ud.roi[2:4] if args.crop_to_roi else (w, h)
        out_path = Path(args.out) if args.out else (ROOT / "ml" / "output" /
                                                    f"undistorted_{v.stem}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, out_size)
        print(f"writing {n} frames → {out_path}")
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(ud(fr))
            if i % 50 == 0:
                print(f"  {i}/{n}", end="\r", flush=True)
        cap.release()
        writer.release()
        print(f"\n  done → {out_path}")
        return

    undist = ud(frame)
    out_path = Path(args.out) if args.out else \
        (ROOT / "ml" / "output" / f"undistort_{v.stem}_f{args.frame}.jpg")
    cv2.imwrite(str(out_path), undist, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
