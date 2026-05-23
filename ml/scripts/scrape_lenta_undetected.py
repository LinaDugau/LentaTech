"""Scrape цен с lenta.com."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"

CATEGORIES = [
    "alkogol-17036",
    "molochnye-produkty-3",
    "napitki-4",
    "sladosti-1028",
    "konservaciya-94",
    "maslo-sousy-specii-20824",
    "kofe-chajj-kakao-242",
    "syry-2",
    "kolbasa-sosiski-754",
    "myaso-i-ptica-136",
    "ovoshchi-frukty-144",
    "makarony-krupy-muka-25",
    "hleb-i-vypechka-165",
    "ryba-ikra-moreprodukty-183",
    "zamorozka-77",
    "sneki-20195",
    "zdorovoe-pitanie-1879",
]


def _parse_jsonld(html: str) -> list[dict]:
    out = []
    for m in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.+?)</script>',
            html, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("@type")
            types = [t] if isinstance(t, str) else (t or [])
            if "Product" not in types:
                continue
            offers = it.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            out.append({
                "name": it.get("name", ""),
                "barcode": str(it.get("gtin13") or it.get("gtin") or ""),
                "price": offers.get("price") or "",
                "url": it.get("url") or "",
                "brand": (it.get("brand", {}) or {}).get("name", "")
                          if isinstance(it.get("brand"), dict) else "",
                "category": it.get("category", ""),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "lenta_catalog_uc.parquet"))
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--show", action="store_true",
                    help="показать окно браузера")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    args = ap.parse_args()

    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
    except ImportError:
        print("Установи: pip install undetected-chromedriver selenium")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    options = uc.ChromeOptions()
    if not args.show:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=ru-RU")

    print("Запуск undetected-chromedriver...")
    driver = uc.Chrome(options=options, version_main=None)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(5)

    try:
        print("Warm-up: lenta.com")
        driver.get("https://lenta.com/")
        time.sleep(5)  # дать Qrator пройти + Angular отрендерить
        title = driver.title
        print(f"  title: {title}")
        if "403" in title:
            print("❌ Qrator всё ещё блокирует.")
            return

        all_urls = []
        for cat in args.categories:
            try:
                driver.get(f"https://lenta.com/catalog/{cat}/")
                time.sleep(3)
                for _ in range(3):
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(1)
                links = driver.execute_script(
                    "return Array.from(new Set(Array.from("
                    "document.querySelectorAll('a[href*=\"/product/\"]'))"
                    ".map(e => e.href)))")
                all_urls.extend(links or [])
                print(f"  {cat}: +{len(links)} → total {len(set(all_urls))}")
                if len(set(all_urls)) >= args.max:
                    break
            except Exception as e:
                print(f"  {cat}: {e}")
                continue

        all_urls = list(dict.fromkeys(all_urls))[: args.max]
        if not all_urls:
            print("Не нашли product URL.")
            return

        rows = []
        for u in tqdm(all_urls):
            try:
                driver.get(u)
                time.sleep(1.5)
                html = driver.page_source
                items = _parse_jsonld(html)
                if not items:
                    info = driver.execute_script("""
                        const g = (s) => document.querySelector(s)?.textContent?.trim() || '';
                        const name = g('h1, [automation-id="product-page-name"]');
                        const price = g('.main-price');
                        const all = document.body.innerText;
                        const m = all.match(/(?:Штрихкод|EAN)\\D{0,5}(\\d{12,13})/i);
                        return { name, price, barcode: m ? m[1] : '' };
                    """)
                    if info and info.get("name"):
                        items = [{
                            "name": info["name"],
                            "barcode": info.get("barcode", ""),
                            "price": str(info.get("price", "")).replace("\xa0", "").replace("₽", "").strip(),
                            "url": u, "brand": "", "category": "",
                        }]
                for it in items:
                    it.setdefault("url", u)
                    rows.append(it)
            except Exception:
                continue
    finally:
        driver.quit()

    if not rows:
        print("Пусто.")
        return
    df = pd.DataFrame(rows)
    df["barcode"] = df["barcode"].astype(str).str.replace(r"\D", "", regex=True)
    df = df.drop_duplicates(subset=["barcode", "name"], keep="first")
    out_path = Path(args.out)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ {len(df)} SKUs → {out_path}")
    with_bc = (df["barcode"].astype(str).str.len() >= 12).sum()
    print(f"   c barcode: {with_bc}")


if __name__ == "__main__":
    main()
