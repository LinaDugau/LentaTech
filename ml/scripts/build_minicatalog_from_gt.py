"""Мини-каталог из GT-csv размеченных видео."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Данные"
DATA_DIR = ROOT / "ml" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog.parquet"))
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_dir in sorted(DATA.glob("*")):
        if not video_dir.is_dir() or video_dir.name == "Unlabeled":
            continue
        gt_csv = video_dir / f"{video_dir.name}.csv"
        if not gt_csv.exists():
            continue
        df = pd.read_csv(gt_csv)
        print(f"  {video_dir.name}: {len(df)} строк")
        for _, r in df.iterrows():
            bc = str(r.get("barcode", "")).strip()
            if bc.endswith(".0"):
                bc = bc[:-2]
            bc = "".join(c for c in bc if c.isdigit())
            if len(bc) < 12:
                continue
            rows.append({
                "name": str(r.get("product_name", "")).strip(),
                "barcode": bc,
                "price": str(r.get("price_default", "")).replace(",", "."),
                "card_price": str(r.get("price_card", "")).replace(",", "."),
                "source": f"gt:{video_dir.name}",
                "category": "",  # GT не содержит явной категории
            })

    if not rows:
        print("❌ GT-csv не найдены или пусты.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["barcode"], keep="first")
    out_path = Path(args.out)
    try:
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False, encoding="utf-8")
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Mini-catalog: {len(df)} уникальных SKU → {out_path}")
    print(f"   c barcode 12-13 знаков: {(df['barcode'].str.len() >= 12).sum()}")
    print("\nSample:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
