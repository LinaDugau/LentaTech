"""Гибридный каталог:"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def normbc(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(".0", "")
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    s = re.sub(r"\D", "", s)
    return s if 12 <= len(s) <= 14 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_hack", default=str(ROOT / "db_hack.csv"))
    ap.add_argument("--scraped", default=str(ROOT / "ml" / "data" / "lenta_catalog_merged.parquet"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"))
    args = ap.parse_args()

    print(f"Loading db_hack: {args.db_hack}")
    db = pd.read_csv(args.db_hack, sep=";", encoding="cp1251", dtype=str,
                     low_memory=False)
    db = db.rename(columns={"fullname": "name", "code": "barcode"})
    db["_bc"] = db["barcode"].apply(normbc)
    db = db[db["_bc"] != ""].copy()
    print(f"  db_hack valid: {len(db)} rows (EAN 12-14 digits)")

    print(f"Loading scraped: {args.scraped}")
    sc = pd.read_parquet(args.scraped)
    sc["_bc"] = sc["barcode"].apply(normbc)
    sc = sc[sc["_bc"] != ""].copy()
    print(f"  scraped: {len(sc)} rows")

    sc_prices = sc.drop_duplicates(subset="_bc", keep="first")
    cols_needed = ["_bc"]
    for c in ("price", "card_price", "source", "url"):
        if c in sc_prices.columns:
            cols_needed.append(c)
    sc_prices = sc_prices[cols_needed]
    sc_prices = sc_prices.rename(columns={
        "price": "scraped_price",
        "card_price": "scraped_card_price",
        "source": "scraped_source",
        "url": "scraped_url",
    })

    merged = db.merge(sc_prices, on="_bc", how="left")
    n_enriched = merged["scraped_price"].notna().sum() if "scraped_price" in merged.columns else 0
    print(f"  enriched with prices: {n_enriched} / {len(merged)} barcodes")

    merged["barcode"] = merged["_bc"]
    merged = merged.drop(columns=["_bc"])
    if "scraped_price" in merged.columns:
        merged = merged.rename(columns={
            "scraped_price": "price",
            "scraped_card_price": "card_price",
            "scraped_source": "source",
            "scraped_url": "url",
        })
    if "source" not in merged.columns:
        merged["source"] = "db_hack"
    else:
        merged["source"] = merged["source"].fillna("db_hack")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"\nSaved → {args.out}")
    print(f"  total rows: {len(merged)}")
    print(f"  cols: {list(merged.columns)}")
    print(f"  rows with prices: {(merged.get('price').notna()).sum() if 'price' in merged.columns else 0}")
    print(f"  rows from db_hack only (no prices): "
          f"{len(merged) - ((merged.get('price').notna()).sum() if 'price' in merged.columns else 0)}")


if __name__ == "__main__":
    main()
