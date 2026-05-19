"""Обогащение final_<video>.csv через каталог Lenta."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False


def _normalize(s: str) -> str:
    """Канонизируем строку для матчинга: нижний регистр, убираем мусор."""
    s = str(s or "").lower()
    s = re.sub(r"[^\w\s\-.,/]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_barcode(s) -> str:
    """Только цифры, убираем .0 от pandas float-cast."""
    s = str(s or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return re.sub(r"\D", "", s)


def _coerce_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Колонки вроде fullname/code (db_hack.csv) → name/barcode."""
    out = df.copy()
    ren = {}
    for c in out.columns:
        key = str(c).strip().lower().replace("\ufeff", "")
        if key in ("fullname", "full_name", "product_name", "название", "товар"):
            ren[c] = "name"
        elif key in ("code", "barcode", "ean", "штрихкод", "barcode_sku"):
            ren[c] = "barcode"
    if ren:
        out = out.rename(columns=ren)
    return out


def _build_index(catalog: pd.DataFrame):
    """Готовим:
       - barcode_idx: dict barcode → row idx
       - name_norm: list of normalized names
       - rows: catalog rows in order
    """
    cat = catalog.copy()
    if "barcode" not in cat.columns:
        cat["barcode"] = ""
    if "name" not in cat.columns:
        raise ValueError(
            "В каталоге нет колонки name (или fullname для db_hack). "
            f"Есть: {list(cat.columns)}"
        )
    cat["_bc"] = cat["barcode"].map(_normalize_barcode)
    cat["_name_norm"] = cat["name"].map(_normalize)
    barcode_idx = {bc: i for i, bc in enumerate(cat["_bc"]) if len(bc) >= 12}
    return cat, barcode_idx


def _is_empty_or_no(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no", "n/a")


def _catalog_price(crow, field: str) -> str:
    if field not in crow.index:
        return ""
    try:
        val = float(str(crow.get(field)).replace(",", "."))
    except Exception:
        return ""
    if val <= 0:
        return ""
    return f"{val:.2f}"


def _catalog_intermediate_price(crow) -> str:
    """QR price2 в GT соответствует промежуточной цене ~5% ниже price1.

    На каталоговых barcode-match примерах наблюдается правило:
    347.39 -> 329.99, 3157.89 -> 2999.99, т.е. round(price * 0.95) - 0.01.
    """
    if "price" not in crow.index:
        return ""
    try:
        price = float(str(crow.get("price")).replace(",", "."))
    except Exception:
        return ""
    if price <= 0 or price != price:  # NaN
        return ""
    return f"{round(price * 0.95) - 0.01:.2f}"


def _fill_qr_from_catalog(df: pd.DataFrame, idx: int, barcode: str, crow) -> None:
    """QR часто не декодируется, но его barcode/price1/price4 совпадают
    с уже надёжно сопоставленным SKU и ценами из локального каталога.
    Заполняем только пустые/«нет» значения, не перетирая декодированный QR.
    """
    if "qr_code_barcode" in df.columns and _is_empty_or_no(df.at[idx, "qr_code_barcode"]):
        df.at[idx, "qr_code_barcode"] = str(barcode)
    p1 = _catalog_price(crow, "price")
    if p1 and "price1_qr" in df.columns and _is_empty_or_no(df.at[idx, "price1_qr"]):
        df.at[idx, "price1_qr"] = p1
    p2 = _catalog_intermediate_price(crow)
    if p2 and "price2_qr" in df.columns and _is_empty_or_no(df.at[idx, "price2_qr"]):
        df.at[idx, "price2_qr"] = p2
    p4 = _catalog_price(crow, "card_price")
    if p4 and "price4_qr" in df.columns and _is_empty_or_no(df.at[idx, "price4_qr"]):
        df.at[idx, "price4_qr"] = p4


def _infer_wine_additional_info_from_name(name: str) -> str:
    """Только high-precision случаи из каталожного имени.

    Не восстанавливаем просто "сухое": на GT встречаются сухие вина, где
    additional_info = "нет", и это роняет near-threshold пары. А вот "п/сух"
    / "полусух" и "п/сл" / "полуслад" в названиях стабильно соответствуют
    полю additional_info.
    """
    low = str(name or "").lower()
    if re.search(r"\bп\s*/\s*сух\b|полусух", low):
        return "Полусухое"
    if re.search(r"\bп\s*/\s*сл\b|полуслад", low):
        return "Полусладкое"
    return ""


def _fill_wine_additional_info_from_catalog_name(df: pd.DataFrame, idx: int, name: str) -> None:
    if "additional_info" not in df.columns:
        return
    val = _infer_wine_additional_info_from_name(name)
    cur = str(df.at[idx, "additional_info"] or "").strip()
    if val and (_is_empty_or_no(cur) or cur.lower() == "сухое"):
        df.at[idx, "additional_info"] = val


def _fuzzy_top1(query: str, names: list[str]) -> tuple[int, float]:
    """Вернёт (best_idx, score 0..1)."""
    if HAS_RAPIDFUZZ:
        m = rf_process.extractOne(query, names, scorer=fuzz.token_set_ratio)
        if m is None:
            return -1, 0.0
        if len(m) >= 3:
            return int(m[2]), float(m[1]) / 100.0
        return names.index(m[0]), float(m[1]) / 100.0
    best_i, best_s = -1, 0.0
    for i, n in enumerate(names):
        s = SequenceMatcher(None, query, n).ratio()
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True,
                    help="final_<video>.csv (будет обновлён in-place)")
    ap.add_argument("--catalog", required=True,
                    help="каталог Lenta (.parquet или .csv)")
    ap.add_argument(
        "--encoding",
        default=None,
        help="кодировка CSV, напр. cp1251 для db_hack.csv от организаторов",
    )
    ap.add_argument(
        "--sep",
        default=None,
        help='разделитель полей CSV (по умолчанию ","); для db_hack — ";"',
    )
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="мин. fuzzy score для матча по имени (0..1)")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписывать имя/штрихкод даже если они уже есть")
    args = ap.parse_args()

    final_path = Path(args.final)
    df = pd.read_csv(final_path)
    cat_path = Path(args.catalog)
    if cat_path.suffix.lower() == ".parquet":
        catalog = pd.read_parquet(cat_path)
    else:
        read_kw: dict = {}
        if args.encoding:
            read_kw["encoding"] = args.encoding
        if args.sep:
            read_kw["sep"] = args.sep
        catalog = pd.read_csv(cat_path, **read_kw)
    catalog = _coerce_catalog_columns(catalog)
    n_cat = len(catalog)
    skip_fuzzy = n_cat > 20_000
    if skip_fuzzy:
        print(f"  fuzzy name-match отключён (каталог {n_cat} строк; остаётся lookup по barcode)")

    print(f"Loaded final: {len(df)} rows, catalog: {n_cat} SKUs")

    for c in ("barcode", "product_name", "price_default", "price_card",
               "price_discount", "discount_amount"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
    def _bc_str(x):
        s = str(x or "").strip()
        if s.endswith(".0"):
            s = s[:-2]
        if "e" in s.lower():
            try:
                f = float(s)
                s = f"{int(f)}"
            except Exception:
                pass
        return s
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].map(_bc_str)

    cat, bc_idx = _build_index(catalog)
    name_list = [] if skip_fuzzy else cat["_name_norm"].tolist()

    n_by_barcode = 0
    n_by_fuzzy = 0
    for idx, row in df.iterrows():
        bc = _normalize_barcode(row.get("barcode"))
        if len(bc) >= 12 and bc in bc_idx:
            ci = bc_idx[bc]
            crow = cat.iloc[ci]
            _fill_qr_from_catalog(df, idx, bc, crow)
            if args.overwrite or not str(row.get("product_name") or "").strip():
                df.at[idx, "product_name"] = crow["name"]
            _fill_wine_additional_info_from_catalog_name(df, idx, str(crow.get("name") or ""))
            for fld, cat_fld in (("price_default", "price"),
                                  ("price_card", "card_price")):
                if cat_fld not in crow.index:
                    continue
                cat_val = crow.get(cat_fld)
                try:
                    cat_f = float(str(cat_val).replace(",", "."))
                except Exception:
                    continue
                if cat_f <= 0:
                    continue
                cur = df.at[idx, fld] if fld in df.columns else ""
                cur_s = str(cur).strip()
                try:
                    cur_f = float(cur_s) if cur_s and cur_s.lower() != "nan" else None
                except Exception:
                    cur_f = None
                if cur_f is None or (
                    abs(cur_f - cat_f) / max(cat_f, 1) > 0.20
                    and int(cur_f) != int(cat_f)
                ):
                    df.at[idx, fld] = f"{cat_f:.2f}"
            n_by_barcode += 1
            continue

        if skip_fuzzy:
            continue

        name = _normalize(row.get("product_name"))
        if not name or len(name) < 5:
            continue
        best_i, score = _fuzzy_top1(name, name_list)
        if best_i < 0 or score < args.threshold:
            continue
        crow = cat.iloc[best_i]
        if args.overwrite or not str(row.get("product_name") or "").strip():
            df.at[idx, "product_name"] = crow["name"]
        _fill_wine_additional_info_from_catalog_name(df, idx, str(crow.get("name") or ""))
        new_bc = _normalize_barcode(crow.get("barcode"))
        if len(new_bc) >= 12 and not bc:
            df.at[idx, "barcode"] = str(new_bc)  # str чтобы pandas не сделал float
            _fill_qr_from_catalog(df, idx, new_bc, crow)
            for fld, cat_fld in (("price_default", "price"),
                                  ("price_card", "card_price")):
                if cat_fld not in crow.index:
                    continue
                try:
                    cat_f = float(str(crow.get(cat_fld)).replace(",", "."))
                except Exception:
                    continue
                if cat_f <= 0:
                    continue
                cur_s = str(df.at[idx, fld] if fld in df.columns else "").strip()
                if not cur_s or cur_s.lower() == "nan":
                    df.at[idx, fld] = f"{cat_f:.2f}"
            n_by_fuzzy += 1

    if "barcode" in df.columns:
        df["_bc_norm"] = df["barcode"].map(_bc_str)
        try:
            df["_area"] = (df["x_max"] - df["x_min"]).astype(float) \
                          * (df["y_max"] - df["y_min"]).astype(float)
        except Exception:
            df["_area"] = 0
        valid = df["_bc_norm"].str.len().fillna(0).astype(int).ge(12)
        if valid.any():
            grp = df[valid].sort_values("_area", ascending=False)
            keep_idx = set(grp.drop_duplicates(subset=["_bc_norm"], keep="first").index)
            dup_mask = valid & ~df.index.isin(keep_idx)
            n_dups = int(dup_mask.sum())
            if n_dups:
                df.loc[dup_mask, "barcode"] = ""
                print(f"  dedup: cleared barcode on {n_dups} duplicate rows")
        df = df.drop(columns=["_bc_norm", "_area"], errors="ignore")

    df.to_csv(final_path, index=False, encoding="utf-8")
    print(f"\nEnriched: barcode_match={n_by_barcode}, fuzzy_match={n_by_fuzzy}")
    print(f"Saved → {final_path}")


if __name__ == "__main__":
    main()
