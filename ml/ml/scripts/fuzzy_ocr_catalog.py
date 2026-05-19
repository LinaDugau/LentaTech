"""R4: fuzzy catalog enrichment по OCR-тексту трека (не QR)."""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "match_to_catalog", ROOT / "ml" / "scripts" / "match_to_catalog.py"
)
_mtc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtc)

from build_final_csv import _ean_checksum_ok  # noqa: E402

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

BEVERAGE_KW = re.compile(
    r"вин|wine|портв|шамп|игрист|коньяк|виски|whisky|vodka|водк|ликер|ликёр|"
    r"brandy|champagne|prosecco|cognac|напиток|безалк|пиво|beer|cider|"
    r"сок\b|juice|sparkling|rom\b|rum\b|gin\b|vermouth",
    re.I,
)

_OCR_STOP = {
    "цена", "цены", "карты", "карта", "карте", "руб", "руб.", "акции", "акция",
    "налог", "ндс", "лента", "новинка", "российское", "российсхое", "нет",
    "сухое", "белое", "красное", "полусухое", "полусладкое", "крепленое",
    "креплёное", "столовое", "vino", "wine",
}

_LAT_RX = re.compile(r"[A-Za-z]{3,}")
_CYR_RX = re.compile(r"[А-Яа-яЁё]{4,}")
_PRICE_RX = re.compile(r"\b(\d{2,4}[.,]\d{2})\b")


def _normbc(x) -> str:
    return _mtc._normalize_barcode(x)


def _empty(x) -> bool:
    return _mtc._is_empty_or_no(x)


def _track_text(final_row, ocr_df: pd.DataFrame) -> str:
    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(str(final_row.get("x_min", 0)).replace(",", "."))
    y0 = float(str(final_row.get("y_min", 0)).replace(",", "."))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
    ]
    if cand.empty:
        return ""
    tid = int(cand.iloc[0]["track_id"]) if "track_id" in cand.columns else None
    sub = ocr_df[ocr_df["track_id"] == tid] if tid is not None else cand
    chunks: list[str] = []
    for col in ("ocr_text", "paddle_text", "bottom_extra_text", "top_extra_text"):
        if col not in sub.columns:
            continue
        for val in sub[col].dropna():
            s = str(val).strip()
            if s and s.lower() != "nan":
                chunks.append(s)
    name = str(final_row.get("product_name", "") or "").strip()
    if name and name.lower() not in ("нет", "nan"):
        chunks.append(name)
    return "\n".join(chunks)


def _ocr_prices(text: str) -> list[float]:
    out: list[float] = []
    for m in _PRICE_RX.findall(text):
        try:
            v = float(m.replace(",", "."))
            if 10 <= v <= 9999:
                out.append(v)
        except Exception:
            pass
    return out


def _raw_tokens(text: str) -> list[str]:
    low = str(text or "").lower()
    toks: list[str] = []
    for t in _LAT_RX.findall(low):
        toks.append(t.lower())
    for t in _CYR_RX.findall(low):
        toks.append(t.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in _OCR_STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _brand_tokens(text: str, brand_vocab: list[str], min_ratio: int = 84) -> list[str]:
    """Fuzzy-map OCR latin fragments → catalog brand tokens (ABRAU, SANTO, …)."""
    if not HAS_RAPIDFUZZ or not brand_vocab:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in _LAT_RX.findall(str(text)):
        w = raw.lower()
        if len(w) < 3 or w in _OCR_STOP:
            continue
        m = rf_process.extractOne(w, brand_vocab, scorer=fuzz.ratio)
        m_part = rf_process.extractOne(w, brand_vocab, scorer=fuzz.partial_ratio)
        best = m if (m and (not m_part or m[1] >= m_part[1])) else m_part
        score = best[1] if best else 0
        if best and score >= min_ratio:
            tok = best[0]
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


def _filter_subset(catalog: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    if not subset or subset == "all":
        return catalog
    if subset == "beverage":
        mask = catalog["name"].fillna("").astype(str).str.contains(BEVERAGE_KW, na=False)
        return catalog[mask].copy()
    raise ValueError(f"unknown subset: {subset}")


def _build_brand_vocab(names: list[str]) -> list[str]:
    vocab: set[str] = set()
    for name in names:
        for t in re.findall(r"[a-z]{4,}", _mtc._normalize(name)):
            if t not in _OCR_STOP:
                vocab.add(t)
    return sorted(vocab)


def _build_token_index(names: list[str]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for i, name in enumerate(names):
        for t in re.findall(r"[a-z]{3,}|[а-яё]{4,}", name):
            index[t].add(i)
    return index


def _catalog_price(row, field: str) -> float | None:
    if field not in row.index:
        return None
    try:
        v = float(str(row.get(field)).replace(",", "."))
    except Exception:
        return None
    return v if v > 0 else None


def process(
    final_path: Path,
    ocr_path: Path,
    catalog_path: Path,
    *,
    subset: str | None = "beverage",
    threshold: float = 0.82,
    brand_ratio: int = 84,
    min_tokens: int = 1,
    encoding: str | None = None,
    sep: str | None = None,
    overwrite: bool = False,
    video: str | None = None,
    min_score: float | None = None,
    require_brand_token: bool = True,
) -> tuple[int, int]:
    if not HAS_RAPIDFUZZ:
        print("  rapidfuzz required")
        return 0, 0

    df = pd.read_csv(final_path)
    ocr = pd.read_csv(ocr_path)

    if catalog_path.suffix.lower() == ".parquet":
        catalog = pd.read_parquet(catalog_path)
    else:
        kw: dict = {}
        if encoding:
            kw["encoding"] = encoding
        if sep:
            kw["sep"] = sep
        catalog = pd.read_csv(catalog_path, **kw)
    catalog = _mtc._coerce_catalog_columns(catalog)
    catalog = _filter_subset(catalog, subset)
    if catalog.empty:
        print("  catalog subset empty")
        return 0, 0

    cat, bc_idx = _mtc._build_index(catalog)
    names_norm = cat["_name_norm"].tolist()
    brand_vocab = _build_brand_vocab(names_norm)
    token_index = _build_token_index(names_norm)

    for c in ("barcode", "product_name", "price_default", "price_card"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")

    n_fuzzy = 0
    n_skip = 0
    for idx, row in df.iterrows():
        bc = _normbc(row.get("barcode"))
        if len(bc) >= 12 and bc in bc_idx:
            n_skip += 1
            continue

        text = _track_text(row, ocr)
        if not text.strip():
            continue

        query_tokens = _raw_tokens(text)
        brand_hits = _brand_tokens(text, brand_vocab, brand_ratio)
        if require_brand_token and not brand_hits:
            continue
        query_tokens.extend(brand_hits)
        query_tokens = list(dict.fromkeys(query_tokens))
        if len(query_tokens) < min_tokens:
            continue

        cands: set[int] = set()
        for t in query_tokens:
            if t in token_index:
                cands |= token_index[t]
        if not cands and query_tokens:
            for t in query_tokens:
                if len(t) >= 4 and t.isascii():
                    for i, name in enumerate(names_norm):
                        if t in name:
                            cands.add(i)
        if not cands:
            continue

        sub_idx = sorted(cands)
        sub_names = [names_norm[i] for i in sub_idx]
        query = " ".join(query_tokens)
        m = rf_process.extractOne(query, sub_names, scorer=fuzz.token_set_ratio)
        score = (m[1] / 100.0) if m else 0.0
        if score < threshold:
            continue
        if min_score is not None and score < min_score:
            continue

        ci = sub_idx[sub_names.index(m[0])]
        crow = cat.iloc[ci]
        new_bc = _normbc(crow.get("barcode"))
        if len(new_bc) < 12 or new_bc not in bc_idx:
            continue
        if len(new_bc) == 13 and not _ean_checksum_ok(new_bc):
            continue

        ocr_prices = _ocr_prices(text)
        cat_p = _catalog_price(crow, "price")
        if ocr_prices and cat_p is not None:
            if not any(abs(p - cat_p) / max(cat_p, 1) <= 0.20 for p in ocr_prices):
                if m[1] < max(threshold, 0.92) * 100:
                    continue

        if overwrite or _empty(row.get("product_name")):
            df.at[idx, "product_name"] = crow["name"]
        if overwrite or len(bc) < 12:
            df.at[idx, "barcode"] = str(new_bc)
        try:
            _mtc._fill_qr_from_catalog(df, idx, new_bc, crow)
        except (ValueError, TypeError):
            if "qr_code_barcode" in df.columns and _empty(row.get("qr_code_barcode")):
                df.at[idx, "qr_code_barcode"] = str(new_bc)
        _mtc._fill_wine_additional_info_from_catalog_name(df, idx, str(crow.get("name") or ""))

        for fld, cat_fld in (("price_default", "price"), ("price_card", "card_price")):
            cp = _catalog_price(crow, cat_fld)
            if cp is None:
                continue
            cur = str(df.at[idx, fld] if fld in df.columns else "").strip()
            if overwrite or _empty(cur):
                df.at[idx, fld] = f"{cp:.2f}"

        n_fuzzy += 1

    if "barcode" in df.columns and n_fuzzy:
        def _bc_str(x):
            s = str(x or "").strip()
            if s.endswith(".0"):
                s = s[:-2]
            return re.sub(r"\D", "", s)

        df["_bc_norm"] = df["barcode"].map(_bc_str)
        try:
            df["_area"] = (df["x_max"] - df["x_min"]).astype(float) * (df["y_max"] - df["y_min"]).astype(float)
        except Exception:
            df["_area"] = 0
        valid = df["_bc_norm"].str.len().fillna(0).astype(int).ge(12)
        if valid.any():
            grp = df[valid].sort_values("_area", ascending=False)
            keep_idx = set(grp.drop_duplicates(subset=["_bc_norm"], keep="first").index)
            dup_mask = valid & ~df.index.isin(keep_idx)
            if dup_mask.any():
                df.loc[dup_mask, "barcode"] = ""
        df = df.drop(columns=["_bc_norm", "_area"], errors="ignore")

    df.to_csv(final_path, index=False, encoding="utf-8")
    tag = video or final_path.stem.replace("final_eval_", "").replace("final_", "")
    print(f"  {tag}: fuzzy_ocr={n_fuzzy}, skipped_has_bc={n_skip}, subset={subset or 'all'}")
    return n_fuzzy, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None)
    ap.add_argument("--ocr", default=None)
    ap.add_argument("--catalog", default=str(ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"))
    ap.add_argument("--subset", default="beverage", choices=("beverage", "all"))
    ap.add_argument("--threshold", type=float, default=0.82)
    ap.add_argument("--brand-ratio", type=int, default=84)
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-require-brand", action="store_true",
                    help="не требовать fuzzy brand-токен в OCR")
    ap.add_argument("--min-score", type=float, default=None,
                    help="доп. нижняя граница score (строже threshold)")
    ap.add_argument("--all", action="store_true", help="все eval-видео в ml/output")
    ap.add_argument("--video-only", default=None,
                    help="применять только к одному видео, напр. 25_12-20")
    ap.add_argument("--video", action="append")
    args = ap.parse_args()

    _kw = dict(
        subset=args.subset, threshold=args.threshold,
        brand_ratio=args.brand_ratio,
        encoding=args.encoding, sep=args.sep,
        overwrite=args.overwrite, min_score=args.min_score,
        require_brand_token=not args.no_require_brand,
    )

    if args.all or args.video or args.video_only:
        videos = [args.video_only] if args.video_only else (
            args.video or ["25_12-20", "26_12-20", "43_15"]
        )
        total = 0
        out = ROOT / "ml" / "output"
        for v in videos:
            final = out / f"final_eval_{v}.csv"
            ocr = out / f"ocr_qr_{v}.csv"
            if not final.is_file():
                final = out / f"final_{v}.csv"
            if not final.is_file() or not ocr.is_file():
                print(f"skip {v}")
                continue
            n, _ = process(
                final, ocr, Path(args.catalog),
                video=v, **_kw,
            )
            total += n
        print(f"Total fuzzy_ocr: {total}")
        return

    if not args.all and not args.video and not args.video_only:
        if not args.final or not args.ocr:
            ap.error("--final and --ocr required unless --all / --video / --video-only")
        process(
            Path(args.final), Path(args.ocr), Path(args.catalog),
            **_kw,
        )


if __name__ == "__main__":
    main()
