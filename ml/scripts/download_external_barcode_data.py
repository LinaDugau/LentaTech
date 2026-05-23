"""Скачивание barcode-датасетов."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "data" / "external_barcode"

# Источники (BenSouchet/barcode-datasets, viplabB/SBD, BarBeR)
SOURCES = {
    "sbd_lr": {
        "url": "https://drive.google.com/uc?id=1q9AU1y9qs2yaAPe1K2a8Th5AFKm03_uZ",
        "file": "BarcodesLR.tar.gz",
        "size": "~2.2 GB",
        "license": "CC BY 4.0",
        "note": "100k synthetic 400x400, EAN/UPC/Code128/QR",
    },
    "barber": {
        "url": "https://ditto.ing.unimore.it/barber/",
        "file": "manual",
        "note": "8748 images — нужна регистрация на сайте; положить dataset/ + Annotations/ в external_barcode/barber/",
    },
    "deal_kaist": {
        "url": "https://www.resl.kaist.ac.kr/doc/datasets",
        "file": "manual",
        "note": "3200 EAN-13, CC BY 3.0 — скачать вручную → external_barcode/deal_kaist/",
    },
    "artelab": {
        "url": "http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html",
        "file": "manual",
        "note": "366 EAN-13 — распаковать в external_barcode/artelab/",
    },
    "muenster": {
        "url": "https://www.uni-muenster.de/PRIA/en/forschung/index.shtml",
        "file": "manual",
        "note": "1055 EAN/UPC — распаковать в external_barcode/muenster/",
    },
}


def _gdown(url: str, dest: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"  already exists: {dest} ({dest.stat().st_size // 1_048_576} MB)")
        return True
    print(f"  downloading → {dest}")
    gdown.download(url, str(dest), quiet=False)
    return dest.is_file()


def main():
    ap = argparse.ArgumentParser(description="Download public barcode datasets")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", choices=list(SOURCES.keys()), action="append")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in SOURCES.items():
            print(f"\n[{k}] {v.get('size', '')}")
            print(f"  URL: {v['url']}")
            print(f"  {v['note']}")
        return

    keys = args.only or ["sbd_lr"]
    args.out.mkdir(parents=True, exist_ok=True)

    for key in keys:
        info = SOURCES[key]
        print(f"\n=== {key} ===")
        print(f"  {info['note']}")
        if info["file"] == "manual":
            print(f"  Manual: {info['url']}")
            continue
        dest = args.out / info["file"]
        ok = _gdown(info["url"], dest)
        print(f"  {'OK' if ok else 'FAILED'}: {dest}")

    readme = args.out / "README_SOURCES.txt"
    lines = ["Public barcode dataset sources for Lenta ML pretrain\n"]
    for k, v in SOURCES.items():
        lines.append(f"{k}: {v['url']}\n  {v['note']}\n")
    readme.write_text("".join(lines), encoding="utf-8")
    print(f"\nSources saved → {readme}")


if __name__ == "__main__":
    main()
