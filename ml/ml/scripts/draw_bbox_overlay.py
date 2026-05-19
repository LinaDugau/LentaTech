"""BBox overlay на кадре видео."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))
from video_resolve import find_video  # noqa: E402

COLORS = [
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (231, 76, 60),    # red
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (149, 165, 166),  # gray
]


def _f(x) -> float:
    if pd.isna(x):
        return 0.0
    return float(str(x).strip().replace(",", "."))


def _load_rows(video: str, csv_path: Path | None, use_gt: bool) -> pd.DataFrame:
    if use_gt or csv_path is None:
        p = ROOT / "Данные" / video / f"{video}.csv"
    else:
        p = csv_path
    if not p.is_file():
        raise SystemExit(f"CSV not found: {p}")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _pick_timestamp(df: pd.DataFrame, ts_ms: float | None) -> float:
    if ts_ms is not None:
        return ts_ms
    df = df.copy()
    df["_ts"] = df["frame_timestamp"].map(_f)
    counts = df.groupby("_ts").size()
    return float(counts.idxmax())


def _get_font(size: int):
    try:
        from PIL import ImageFont
        for name in (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "Arial.ttf",
        ):
            p = Path(name)
            if p.is_file():
                return ImageFont.truetype(str(p), size)
        return ImageFont.load_default()
    except Exception:
        return None


def _put_label_pil(img: np.ndarray, text: str, x: int, y: int, color_bgr: tuple) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        cv2.putText(img, text.encode("ascii", "replace").decode(), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        return img
    font = _get_font(22)
    if font is None:
        return img
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=rgb)
    draw.text((x + pad, y + pad), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _short_label(row) -> str:
    name = str(row.get("product_name", "") or "").strip()
    bc = str(row.get("barcode", "") or "").strip()
    if bc and bc.lower() not in ("нет", "nan") and bc.replace(".", "").isdigit():
        bc = bc.split(".")[0]
        tail = bc[-4:]
        if name and name.lower() != "нет":
            words = name.replace(",", " ").split()
            short = " ".join(words[:4])
            if len(short) > 36:
                short = short[:33] + "…"
            return f"{short} · …{tail}"
        return f"EAN …{tail}"
    if name and name.lower() not in ("нет", "nan", ""):
        return name[:36] + ("…" if len(name) > 36 else "")
    return "ценник"


def _read_frame(vpath: Path, ts_ms: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {vpath}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    idx = int(round(ts_ms * fps / 1000.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Cannot read frame at {ts_ms} ms (idx {idx})")
    return frame


def _draw_overlay(
    frame: np.ndarray,
    rows: pd.DataFrame,
    ts_ms: float,
    *,
    tol_ms: float = 120.0,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    tol = tol_ms
    sub = rows[rows["frame_timestamp"].map(_f).sub(ts_ms).abs() <= tol]
    if sub.empty:
        ts_vals = rows["frame_timestamp"].map(_f)
        nearest = float(ts_vals.iloc[(ts_vals - ts_ms).abs().argmin()])
        sub = rows[ts_vals.sub(nearest).abs() <= tol]
        ts_ms = nearest
    if sub.empty:
        raise SystemExit(f"No rows within {tol} ms of timestamp {ts_ms}")

    cv2.rectangle(out, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.putText(
        out,
        f"ShelfVision | {rows.iloc[0].get('filename', 'video')} | t={ts_ms:.0f} ms | n={len(sub)}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i, (_, row) in enumerate(sub.iterrows()):
        x0 = int(_f(row.get("x_min", 0)))
        y0 = int(_f(row.get("y_min", 0)))
        x1 = int(_f(row.get("x_max", 0)))
        y1 = int(_f(row.get("y_max", 0)))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 4)
        label = _short_label(row)
        ty0 = max(0, y0 - 40)
        out = _put_label_pil(out, label, x0, ty0, color)

    leg_y = h - 36
    cv2.rectangle(out, (0, leg_y - 28), (min(w, 520), h), (20, 20, 20), -1)
    cv2.putText(
        out,
        "bbox = detected pricetag | label = product + barcode tail",
        (24, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out, ts_ms, len(sub)


def _resize_for_slide(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    nh = int(h * max_w / w)
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="BBox overlay for presentation")
    ap.add_argument("--video", default="26_12-20")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--gt", action="store_true", help="use GT CSV from Данные/")
    ap.add_argument("--timestamp", type=float, default=None, help="frame_timestamp ms")
    ap.add_argument("--tol-ms", type=float, default=120.0)
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    video = args.video
    rows = _load_rows(video, args.csv, args.gt or args.csv is None)
    ts = _pick_timestamp(rows, args.timestamp)

    vname = str(rows.iloc[0].get("filename", f"{video}.mp4"))
    vpath = find_video(vname) or find_video(f"{video}.mp4")
    if vpath is None:
        raise SystemExit(f"Video not found for {video}")

    frame = _read_frame(vpath, ts)
    overlay, ts_used, n_tags = _draw_overlay(frame, rows, ts, tol_ms=args.tol_ms)
    overlay = _resize_for_slide(overlay, args.max_width)

    suffix = "gt" if (args.gt or args.csv is None) else "pred"
    out = args.out or (ROOT / "docs" / "presentation" / f"bbox_overlay_{video}_{suffix}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved {out} ({overlay.shape[1]}x{overlay.shape[0]}) ts={ts_used:.0f} ms tags={n_tags}")


if __name__ == "__main__":
    main()
