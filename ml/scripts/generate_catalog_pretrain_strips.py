"""Синтетические digit-strip из реальных EAN каталога (db_hack) + python-barcode."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import (  # noqa: E402
    STRIP_H,
    STRIP_W,
    augment_strip,
    ean_checksum_ok,
    extract_digit_strip,
    normalize_barcode,
    render_synthetic_strip,
)

DEFAULT_CATALOG = ROOT / "db_hack.csv"
OUT_DIR = ROOT / "ml" / "data" / "barcode_strips_pretrain"


def _load_catalog_eans(path: Path, limit: int) -> list[str]:
    import pandas as pd

    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            df = pd.read_csv(
                path, dtype=str, keep_default_na=False, sep=";",
                encoding=enc, nrows=limit * 5 if limit else None,
            )
            break
        except UnicodeDecodeError:
            df = None
    if df is None:
        raise SystemExit(f"Cannot read catalog: {path}")
    col = "code" if "code" in df.columns else ("barcode" if "barcode" in df.columns else df.columns[-1])
    seen: set[str] = set()
    out: list[str] = []
    for raw in df[col].astype(str):
        bc = normalize_barcode(raw)
        if len(bc) == 12:
            d = [int(c) for c in bc]
            check = (10 - ((sum(d[0:11:2]) + 3 * sum(d[1:11:2])) % 10)) % 10
            bc = bc + str(check)
        if len(bc) != 13 or not ean_checksum_ok(bc):
            continue
        if bc in seen:
            continue
        seen.add(bc)
        out.append(bc)
        if limit and len(out) >= limit:
            break
    return out


def _render_ean_tag_image(code: str) -> np.ndarray | None:
    """EAN-13 PNG → ценникоподобный crop с полосой цифр снизу."""
    try:
        from barcode import EAN13
        from barcode.writer import ImageWriter
    except ImportError:
        return None

    import io
    from PIL import Image

    buf = io.BytesIO()
    EAN13(code[:12], writer=ImageWriter()).write(buf, {"write_text": True, "font_size": 14, "text_distance": 2})
    buf.seek(0)
    pil = Image.open(buf).convert("L")
    arr = np.array(pil)
    # digit strip ≈ нижняя треть изображения штрихкода
    h, w = arr.shape[:2]
    y0 = int(h * 0.72)
    strip = arr[y0:h, :]
    if strip.size == 0:
        return None
    strip = cv2.resize(strip, (STRIP_W, STRIP_H), interpolation=cv2.INTER_AREA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=8000, help="unique EAN count")
    ap.add_argument("--aug", type=int, default=3, help="augmentations per EAN")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.catalog.is_file():
        raise SystemExit(f"Catalog not found: {args.catalog}")

    eans = _load_catalog_eans(args.catalog, args.limit)
    if not eans:
        raise SystemExit("No valid EAN-13 in catalog")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for i, code in enumerate(eans):
        variants: list[np.ndarray] = []
        tag = _render_ean_tag_image(code)
        if tag is not None:
            variants.append(tag)
        variants.append(render_synthetic_strip(code, rng))
        for _ in range(args.aug):
            base = variants[rng.integers(0, len(variants))]
            variants.append(augment_strip(base, rng))

        for j, strip in enumerate(variants):
            sid = f"cat_{i:05d}_{j:02d}"
            rel = f"strips/{sid}.png"
            cv2.imwrite(str(args.out / rel), strip)
            rows.append({
                "id": sid,
                "source": "catalog_synth",
                "video": "",
                "barcode": code,
                "strip_path": rel,
            })

    labels = args.out / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "video", "barcode", "strip_path"])
        w.writeheader()
        w.writerows(rows)

    print(f"Generated {len(rows)} pretrain strips from {len(eans)} EAN → {args.out}")


if __name__ == "__main__":
    main()
