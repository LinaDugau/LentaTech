"""Разведка видео и GT."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
OUT = ROOT / "ml" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": str(path), "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "frames": n,
        "duration_s": round(n / fps, 2) if fps else None,
    }


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gt_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        if c in df.columns:
            df[c] = df[c].map(to_float)
    df["w"] = df["x_max"] - df["x_min"]
    df["h"] = df["y_max"] - df["y_min"]
    return {
        "csv": csv_path.relative_to(ROOT).as_posix(),
        "rows": len(df),
        "unique_barcodes": df["barcode"].nunique(dropna=True),
        "colors": df["color"].value_counts(dropna=False).to_dict(),
        "bbox_w_mean": round(df["w"].mean(), 1),
        "bbox_h_mean": round(df["h"].mean(), 1),
        "bbox_w_min": round(df["w"].min(), 1),
        "bbox_h_min": round(df["h"].min(), 1),
        "ts_min_ms": df["frame_timestamp"].min(),
        "ts_max_ms": df["frame_timestamp"].max(),
        "missing_pct": (df.isin(["нет"]).sum() / len(df) * 100).round(1).to_dict(),
    }


def main():
    rows = []
    for mp4 in sorted(DATA.rglob("*.mp4")):
        rows.append(video_info(mp4))
    vdf = pd.DataFrame(rows)
    print("\n=== VIDEO META ===")
    print(vdf.to_string(index=False))
    vdf.to_csv(OUT / "video_meta.csv", index=False)

    print("\n=== GT STATS ===")
    for csv in sorted(DATA.rglob("*.csv")):
        s = gt_stats(csv)
        print(f"\n-- {s['csv']}")
        for k, v in s.items():
            if k == "csv":
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
