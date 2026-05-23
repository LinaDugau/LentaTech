"""Decode нижней зоны штрихкода."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from apply_undistort import UndistortMap  # noqa: E402
from barcode_recovery_probe import decode_image  # noqa: E402
from build_final_csv import choose_best_barcode, _ean_checksum_ok  # noqa: E402
from extra_top_ocr import _VideoReader, _crop_from_row, _pick_track_rows, _blur_var  # noqa: E402
from video_resolve import find_video  # noqa: E402
from whole_frame_codes2 import _propagate_tracks  # noqa: E402


def _load_catalog(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    out: set[str] = set()
    if path.suffix.lower() == ".parquet":
        cat = pd.read_parquet(path)
        col = next((c for c in ("code", "barcode", "ean") if c in cat.columns), None)
        if col:
            for v in cat[col].dropna():
                s = re.sub(r"\D", "", str(v))
                if len(s) >= 12:
                    out.add(s[:13])
                    out.add(s[:12])
        return out
    import csv
    with path.open(encoding="cp1251", errors="replace") as f:
        r = csv.reader(f, delimiter=";")
        hdr = next(r, None)
        if not hdr:
            return out
        ci = next((i for i, c in enumerate(hdr) if str(c).strip().lower() == "code"), 1)
        for row in r:
            if ci < len(row):
                s = re.sub(r"\D", "", row[ci])
                if len(s) >= 12:
                    out.add(s[:13])
                    out.add(s[:12])
    return out


def _unsharp(gray: np.ndarray, amount: float = 1.3) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _barcode_zones(crop: np.ndarray) -> list[np.ndarray]:
    """Несколько нижних зон ценника под штрихкод + human-readable digits."""
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    zones: list[np.ndarray] = []
    specs = (
        (0.42, 0.98, 0.02, 0.98),
        (0.50, 1.00, 0.05, 0.95),
        (0.55, 0.95, 0.08, 0.92),
        (0.38, 0.88, 0.00, 1.00),
    )
    for y0f, y1f, x0f, x1f in specs:
        y0 = max(0, int(h * y0f))
        y1 = min(h, max(y0 + 8, int(h * y1f)))
        x0 = max(0, int(w * x0f))
        x1 = min(w, max(x0 + 8, int(w * x1f)))
        z = crop[y0:y1, x0:x1]
        if z.size:
            zones.append(z)
    zones.append(crop)
    return zones


def _preprocess_variants(zone: np.ndarray) -> list[np.ndarray]:
    if zone is None or zone.size == 0:
        return []
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if zone.ndim == 3 else zone
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(gray)
    sharp = _unsharp(clahe)
    den = cv2.bilateralFilter(sharp, d=5, sigmaColor=40, sigmaSpace=40)
    out: list[np.ndarray] = []
    for base, scale in ((sharp, 3.0), (den, 3.5), (clahe, 4.0), (sharp, 4.5)):
        up = cv2.resize(
            base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4,
        )
        if max(up.shape) > 2400:
            f = 2400 / max(up.shape)
            up = cv2.resize(up, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
    return out


def decode_barcode_zone(crop: np.ndarray, *, fast: bool = False) -> Counter[str]:
    hits: Counter[str] = Counter()
    zones = _barcode_zones(crop)
    if fast:
        zones = zones[:2] if zones else []
    for zone in zones:
        procs = _preprocess_variants(zone) if not fast else [
            cv2.cvtColor(
                cv2.resize(
                    cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if zone.ndim == 3 else zone,
                    None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4,
                ),
                cv2.COLOR_GRAY2BGR,
            )
        ]
        for proc in procs:
            for code in decode_image(proc, fast=True):
                if re.fullmatch(r"\d{12,13}", code):
                    hits[code[:13] if len(code) >= 13 else code] += 1
        if not fast:
            for code in decode_image(zone, fast=True):
                if re.fullmatch(r"\d{12,13}", code):
                    hits[code[:13] if len(code) >= 13 else code] += 1
    return hits


def _row_crop(row, reader: _VideoReader | None, undistort: UndistortMap | None) -> np.ndarray | None:
    cp = row.get("hires_crop") or row.get("crop_path")
    if isinstance(cp, str) and cp.strip():
        p = ROOT / cp if not Path(cp).is_absolute() else Path(cp)
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None and img.size:
                return img
    if reader is not None:
        return _tag_crop(reader, row, undistort)
    return _crop_from_row(row, reader)


def _pick_barcode(hits: Counter[str], catalog: set[str], require_catalog: bool) -> str:
    if not hits:
        return ""
    ranked = sorted(
        hits.keys(),
        key=lambda c: (
            hits[c],
            int(_ean_checksum_ok(c[:13])),
            int(c in catalog or c[:12] in catalog),
            len(c),
        ),
        reverse=True,
    )
    for code in ranked:
        c13 = code[:13]
        if len(c13) == 13 and _ean_checksum_ok(c13):
            if not require_catalog or c13 in catalog or code[:12] in catalog:
                return c13
        elif len(code) == 12 and not require_catalog:
            return code
    return ""


def _tag_crop(reader: _VideoReader, row, undistort: UndistortMap | None) -> np.ndarray | None:
    frame = reader.frame_at_ms(float(row["frame_ts_ms"]))
    if frame is None:
        return _crop_from_row(row, None)
    if undistort is not None:
        frame = undistort(frame)
    x1 = int(float(row["x_min_orig"]))
    y1 = int(float(row["y_min_orig"]))
    x2 = int(float(row["x_max_orig"]))
    y2 = int(float(row["y_max_orig"]))
    pad_x = int(0.06 * (x2 - x1))
    pad_y = int(0.04 * (y2 - y1))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(reader.w - 1, x2 + pad_x)
    y2 = min(reader.h - 1, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _pick_frames(sub: pd.DataFrame, reader: _VideoReader | None, top_k: int, all_frames: bool) -> pd.DataFrame:
    if all_frames:
        return sub
    if reader is None:
        return sub.drop_duplicates("track_id", keep="first").head(top_k)
    scored: list[tuple[float, int]] = []
    for idx, row in sub.iterrows():
        crop = _row_crop(row, None, None)
        if crop is None or crop.size == 0:
            continue
        zone = _barcode_zones(crop)
        if not zone:
            continue
        scored.append((_blur_var(zone[0]), int(idx)))
    scored.sort(reverse=True)
    if not scored:
        return _pick_track_rows(sub, sharpest=True, reader=reader).head(top_k)
    keep = [i for _, i in scored[:top_k]]
    return sub.loc[keep]


def process_file(
    csv_path: Path,
    catalog: set[str],
    *,
    force: bool = False,
    top_k: int = 5,
    all_frames: bool = False,
    fast: bool = False,
    require_catalog: bool = True,
    undistort: bool = False,
    crop_only: bool = False,
) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    if df.empty or "track_id" not in df.columns:
        return 0, 0
    if "barcode_raw" not in df.columns:
        df["barcode_raw"] = ""
    df["barcode_raw"] = df["barcode_raw"].astype(object)

    video_name = str(df.iloc[0].get("video", "") or "").strip()
    reader: _VideoReader | None = None
    ud: UndistortMap | None = None
    if not crop_only:
        vpath = find_video(video_name)
        if vpath is None:
            print(f"  video not found: {video_name}, falling back to crop files")
            crop_only = True
        else:
            reader = _VideoReader(vpath)
            if undistort:
                ud = UndistortMap(reader.w, reader.h, crop_to_roi=False)
            print(f"  {csv_path.name}: {vpath.name} ({reader.w}x{reader.h}) undistort={undistort}")
    if crop_only:
        print(f"  {csv_path.name}: crop-only mode (hires_crop on disk)")

    n_decode = 0
    n_updated = 0
    try:
        for tid, sub in tqdm(list(df.groupby("track_id")), desc=csv_path.stem):
            existing = choose_best_barcode(
                [str(x) for x in sub["barcode_raw"] if str(x).strip() not in ("", "nan")]
            )
            ex = re.sub(r"\D", "", existing)
            if not force and len(ex) >= 12 and (len(ex) == 12 or _ean_checksum_ok(ex[:13])):
                continue

            hits: Counter[str] = Counter()
            rows = _pick_frames(sub, reader, top_k=top_k, all_frames=all_frames)
            for _, row in rows.iterrows():
                crop = _row_crop(row, reader, ud)
                hits += decode_barcode_zone(crop, fast=fast)

            bc = _pick_barcode(hits, catalog, require_catalog)
            if hits:
                n_decode += 1
            if bc:
                df.loc[df["track_id"] == tid, "barcode_raw"] = bc
                n_updated += 1
    finally:
        if reader is not None:
            reader.close()

    _propagate_tracks(df)
    df.to_csv(csv_path, index=False)
    print(f"  decode_any={n_decode}, updated={n_updated}")
    return n_decode, n_updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--catalog", type=Path, default=ROOT / "db_hack.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--undistort", action="store_true")
    ap.add_argument("--crop-only", action="store_true", help="только hires_crop с диска, без 4K video")
    ap.add_argument("--no-require-catalog", action="store_true")
    args = ap.parse_args()

    catalog = _load_catalog(args.catalog)
    print(f"catalog: {len(catalog)} codes")
    files = [Path(p) for p in args.files] if args.files else [
        ROOT / "ml" / "output" / f"ocr_qr_{v}.csv" for v in ("26_12-20", "25_12-20", "43_15")
    ]
    total = 0
    for f in files:
        if not f.is_file():
            print(f"skip missing {f}")
            continue
        print(f"\n=== {f.name} ===")
        _, upd = process_file(
            f, catalog,
            force=args.force,
            top_k=args.top_k,
            all_frames=args.all_frames,
            fast=args.fast,
            require_catalog=not args.no_require_catalog,
            undistort=args.undistort,
            crop_only=args.crop_only,
        )
        total += upd
    print(f"\nTotal tracks updated: {total}")


if __name__ == "__main__":
    main()
