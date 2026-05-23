"""Синтетический датасет из шаблонов."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def _apply_motion_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    kernel = np.zeros((ksize, ksize))
    angle = random.uniform(0, 180)
    angle_rad = np.deg2rad(angle)
    cx, cy = ksize // 2, ksize // 2
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    for i in range(ksize):
        offset = i - ksize // 2
        x = int(cx + offset * dx)
        y = int(cy + offset * dy)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1
    kernel /= max(kernel.sum(), 1)
    return cv2.filter2D(img, -1, kernel)


def _apply_glare(img: np.ndarray) -> np.ndarray:
    """Случайный круглый блик."""
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = random.randint(0, w), random.randint(0, h)
    radius = random.randint(min(h, w) // 5, min(h, w) // 2)
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    alpha = random.uniform(0.15, 0.35)
    return cv2.addWeighted(img.astype(np.float32), 1.0,
                            overlay, alpha, 0).clip(0, 255).astype(np.uint8)


def _apply_perspective(img: np.ndarray, max_warp: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    """Случайный perspective. Возвращает (warped, M)."""
    h, w = img.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + np.random.uniform(-max_warp, max_warp,
                                    size=(4, 2)) * [w, h]
    dst = dst.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
    return warped, M


def _paste_rgba_onto(bg: np.ndarray, fg: np.ndarray,
                      x: int, y: int) -> np.ndarray:
    """Pastes RGBA fg onto BGR bg at (x, y) with alpha-blend."""
    h, w = fg.shape[:2]
    H, W = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return bg
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    region_bg = bg[y1:y2, x1:x2]
    fg_crop = fg[fy1:fy2, fx1:fx2]
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[..., 3:] / 255.0
        rgb = fg_crop[..., :3]
        blended = (rgb.astype(np.float32) * alpha
                   + region_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        bg[y1:y2, x1:x2] = blended
    else:
        bg[y1:y2, x1:x2] = fg_crop[..., :3]
    return bg


def _tight_bbox_from_alpha(rgba: np.ndarray):
    """Узкий bbox по непустой alpha."""
    if rgba.shape[2] != 4:
        h, w = rgba.shape[:2]
        return 0, 0, w, h
    a = rgba[..., 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _augment_template(tpl: np.ndarray,
                       target_h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Возвращает (aug RGBA или RGB, tight_bbox в координатах aug)."""
    h, w = tpl.shape[:2]
    scale = target_h / h
    tpl = cv2.resize(tpl, (max(1, int(w * scale)), target_h),
                      interpolation=cv2.INTER_AREA)
    if tpl.shape[2] == 3:
        a = np.full((tpl.shape[0], tpl.shape[1], 1), 255, dtype=np.uint8)
        tpl = np.concatenate([tpl, a], axis=2)
    angle = random.uniform(-12, 12)
    h2, w2 = tpl.shape[:2]
    M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), angle, 1.0)
    tpl = cv2.warpAffine(tpl, M, (w2, h2),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))
    if random.random() < 0.7:
        tpl, _ = _apply_perspective(tpl, max_warp=0.12)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(tpl[..., :3], cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.2), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.2), 0, 255)
        tpl[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.5:
        rgb = _apply_motion_blur(tpl[..., :3], ksize=random.choice([3, 5, 7]))
        tpl[..., :3] = rgb
    if random.random() < 0.4:
        enc = cv2.imencode(".jpg", tpl[..., :3],
                            [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 75)])[1]
        tpl[..., :3] = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    bbox = _tight_bbox_from_alpha(tpl)
    return tpl, bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=str(ROOT / "Материалы" / "Ценники пнг"))
    ap.add_argument("--backgrounds",
                    default=str(ROOT / "ml" / "output" / "synth" / "backgrounds"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "output" / "yolo_dataset_synth"))
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--templates_per_image", type=str, default="1-4",
                    help="диапазон количества шаблонов на одном фоне")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)

    tpls_dir = Path(args.templates)
    bgs_dir = Path(args.backgrounds)
    out_dir = Path(args.out)
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    template_paths = sorted(tpls_dir.glob("*.png"))
    bg_paths = sorted(bgs_dir.glob("*.jpg"))
    print(f"Templates: {len(template_paths)}  Backgrounds: {len(bg_paths)}")
    if not template_paths or not bg_paths:
        print("❌ Шаблонов или фонов нет. Сначала запусти extract_backgrounds.py")
        return

    tpls = []
    for p in template_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            tpls.append(img)
    print(f"Loaded templates: {len(tpls)}")

    lo, hi = (int(x) for x in args.templates_per_image.split("-"))

    n_train = int(args.count * (1 - args.val_ratio))
    for idx in tqdm(range(args.count), desc="synth"):
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        canvas = bg.copy()

        n_paste = random.randint(lo, hi)
        boxes = []  # (x1, y1, x2, y2) на canvas
        for _ in range(n_paste):
            tpl = random.choice(tpls)
            target_h = random.randint(int(H * 0.06), int(H * 0.30))
            aug, tb = _augment_template(tpl, target_h)
            ah, aw = aug.shape[:2]
            tx, ty = random.randint(-aw // 4, max(1, W - aw // 2)), \
                     random.randint(-ah // 4, max(1, H - ah // 2))
            canvas = _paste_rgba_onto(canvas, aug, tx, ty)
            ax1, ay1, ax2, ay2 = tb
            x1 = max(0, tx + ax1); y1 = max(0, ty + ay1)
            x2 = min(W - 1, tx + ax2); y2 = min(H - 1, ty + ay2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            ok = True
            for bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(x1, bx1), max(y1, by1)
                ix2, iy2 = min(x2, bx2), min(y2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    iw, ih = ix2 - ix1, iy2 - iy1
                    if iw * ih > 0.3 * min((x2-x1)*(y2-y1), (bx2-bx1)*(by2-by1)):
                        ok = False
                        break
            if not ok:
                continue
            boxes.append((x1, y1, x2, y2))

        if random.random() < 0.3:
            ks = random.choice([3, 5])
            canvas = cv2.GaussianBlur(canvas, (ks, ks), 0)
        if random.random() < 0.3:
            canvas = _apply_glare(canvas)

        split = "train" if idx < n_train else "val"
        stem = f"synth_{idx:06d}"
        img_path = out_dir / "images" / split / f"{stem}.jpg"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with lbl_path.open("w") as f:
            for x1, y1, x2, y2 in boxes:
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: price_tag\n", encoding="utf-8")
    print(f"\n✅ Synth dataset → {out_dir}")
    print(f"   train: {len(list((out_dir/'images/train').glob('*.jpg')))}")
    print(f"   val:   {len(list((out_dir/'images/val').glob('*.jpg')))}")


if __name__ == "__main__":
    main()
