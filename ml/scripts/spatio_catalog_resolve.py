"""Catalog match без barcode."""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "match_to_catalog", ROOT / "ml" / "scripts" / "match_to_catalog.py"
)
_mtc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtc)

_spec2 = importlib.util.spec_from_file_location(
    "fuzzy_ocr_catalog", ROOT / "ml" / "scripts" / "fuzzy_ocr_catalog.py"
)
_foc = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_foc)

from build_final_csv import _ean_checksum_ok  # noqa: E402

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _is_placeholder_bc(bc: str) -> bool:
    """Внутренние/мусорные EAN из каталога (2099999*, 2999999*)."""
    if len(bc) < 12:
        return True
    if bc.startswith(("2099999", "2999999", "0000000")):
        return True
    return False


def _normbc(x) -> str:
    s = re.sub(r"\D", "", str(x or ""))
    return s[:13] if len(s) >= 12 else ""


def _empty_bc(x) -> bool:
    return len(_normbc(x)) < 12


def _load_catalog(catalog_path: Path, subset: str | None) -> tuple[pd.DataFrame, dict]:
    if catalog_path.suffix.lower() == ".parquet":
        cat = pd.read_parquet(catalog_path)
    else:
        cat = pd.read_csv(catalog_path, sep=";", encoding="cp1251", on_bad_lines="skip")
    cat = _mtc._coerce_catalog_columns(cat)
    cat = _foc._filter_subset(cat, subset)
    cat, bc_idx = _mtc._build_index(cat)
    return cat, bc_idx


def _price_ok(ocr_text: str, crow, *, tol: float = 0.18) -> bool:
    ocr_prices = _foc._ocr_prices(ocr_text)
    if not ocr_prices:
        return True
    cat_p = _foc._catalog_price(crow, "price")
    if cat_p is None:
        return True
    return any(abs(p - cat_p) / max(cat_p, 1) <= tol for p in ocr_prices)


def _candidates(
    text: str,
    cat: pd.DataFrame,
    bc_idx: dict,
    token_index: dict[str, set[int]],
    brand_vocab: list[str],
    *,
    limit: int = 8,
    min_score: float = 0.72,
) -> list[tuple[int, float]]:
    if not HAS_RAPIDFUZZ or not text.strip():
        return []
    query_tokens = _foc._raw_tokens(text)
    brand_hits = _foc._brand_tokens(text, brand_vocab, min_ratio=82)
    query_tokens.extend(brand_hits)
    query_tokens = list(dict.fromkeys(query_tokens))
    cands: set[int] = set()
    for t in query_tokens:
        if t in token_index:
            cands |= token_index[t]
    if not cands:
        for t in query_tokens:
            if len(t) >= 4 and t.isascii():
                for i, name in enumerate(cat["_name_norm"].tolist()):
                    if t in name:
                        cands.add(i)
    if not cands:
        return []
    sub_idx = sorted(cands)
    sub_names = [cat["_name_norm"].iloc[i] for i in sub_idx]
    query = " ".join(query_tokens)
    hits = rf_process.extract(query, sub_names, scorer=fuzz.token_set_ratio, limit=limit * 2)
    out: list[tuple[int, float]] = []
    seen: set[str] = set()
    for name_norm, score, j in hits:
        if score / 100.0 < min_score:
            continue
        ci = sub_idx[j]
        bc = _normbc(cat.iloc[ci].get("barcode"))
        if _is_placeholder_bc(bc) or bc in seen or bc not in bc_idx:
            continue
        seen.add(bc)
        out.append((ci, score / 100.0))
        if len(out) >= limit:
            break
    return out


def _video_seen(df: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for _, row in df.iterrows():
        bc = _normbc(row.get("barcode"))
        if len(bc) >= 12:
            name = str(row.get("product_name", "") or "").strip()
            seen.append(f"{bc} {name[:60]}")
    return seen[:40]


def _shelf_context(df: pd.DataFrame, idx: int, seen: list[str], max_items: int = 10) -> str:
    rows: list[str] = []
    ts = float(df.at[idx, "frame_timestamp"])
    for j, row in df.iterrows():
        if j == idx:
            continue
        if abs(float(row.get("frame_timestamp", 0)) - ts) > 6000:
            continue
        name = str(row.get("product_name", "") or "").strip()
        bc = _normbc(row.get("barcode"))
        if name and name.lower() not in ("нет", "nan"):
            rows.append(f"- {name}" + (f" EAN={bc}" if bc else ""))
        if len(rows) >= max_items:
            break
    if seen:
        rows.append("Уже в видео: " + "; ".join(seen[:12]))
    return "\n".join(rows) if rows else "—"


def _llm_pick(ocr_text: str, options: list[dict], shelf: str, llm) -> int | None:
    if not options:
        return None
    opts = "\n".join(
        f"{i}. EAN={o['barcode']} | {o['name'][:100]} (score={o['score']:.2f})"
        for i, o in enumerate(options)
    )
    prompt = (
        "Сопоставь шумный OCR ценника с ОДНИМ товаром из списка.\n"
        "Ответ: индекс 0..N-1 или NONE. Не выдумывай EAN.\n"
        "Учитывай бренд, объём, тип напитка; OCR часто с ошибками.\n\n"
        f"OCR:\n{ocr_text[:900]}\n\n"
        f"Контекст полки:\n{shelf}\n\n"
        f"Кандидаты:\n{opts}\n\n"
        "Ответ:"
    )
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=12,
        )
        ans = out["choices"][0]["message"]["content"].strip().upper()
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None
    if "NONE" in ans:
        return None
    m = re.search(r"\b(\d+)\b", ans)
    if not m:
        return None
    pick = int(m.group(1))
    return pick if 0 <= pick < len(options) else None


def resolve_file(
    final_path: Path,
    ocr_path: Path,
    catalog_path: Path,
    *,
    subset: str | None = "beverage",
    auto_threshold: float = 0.90,
    llm_threshold: float = 0.72,
    use_llm: bool = True,
    price_tol: float = 0.18,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    if not HAS_RAPIDFUZZ:
        print("  rapidfuzz required")
        return 0, 0, 0
    if not final_path.is_file() or not ocr_path.is_file():
        return 0, 0, 0

    df = pd.read_csv(final_path, dtype={"barcode": str}, keep_default_na=False)
    ocr = pd.read_csv(ocr_path)
    cat, bc_idx = _load_catalog(catalog_path, subset)
    names_norm = cat["_name_norm"].tolist()
    brand_vocab = _foc._build_brand_vocab(names_norm)
    token_index = _foc._build_token_index(names_norm)
    print(f"  catalog {len(cat)} SKU ({subset or 'all'})")

    llm = None
    if use_llm:
        from llm_product_name import get_llm
        llm = get_llm()

    seen_video = _video_seen(df)
    n_auto = n_llm = n_skip = 0

    for idx, row in tqdm(list(df.iterrows()), desc=final_path.stem):
        if not _empty_bc(row.get("barcode")):
            continue
        text = _foc._track_text(row, ocr)
        if len(text.strip()) < 10:
            n_skip += 1
            continue
        raw_cands = _candidates(
            text, cat, bc_idx, token_index, brand_vocab,
            limit=8, min_score=llm_threshold,
        )
        if not raw_cands:
            n_skip += 1
            continue
        priced: list[dict] = []
        for ci, score in raw_cands:
            crow = cat.iloc[ci]
            if not _price_ok(text, crow, tol=price_tol):
                continue
            bc = _normbc(crow.get("barcode"))
            if _is_placeholder_bc(bc) or len(bc) != 13 or not _ean_checksum_ok(bc) or bc not in bc_idx:
                continue
            priced.append({
                "barcode": bc,
                "name": str(crow.get("name", "") or ""),
                "score": score,
            })
        if not priced:
            n_skip += 1
            continue

        pick: dict | None = None
        best = priced[0]
        # score=1.0 часто артефакт token_set на коротком query — требуем WRatio
        wr = fuzz.WRatio(_mtc._normalize(text), _mtc._normalize(best["name"])) / 100.0
        if best["score"] >= auto_threshold and wr >= max(auto_threshold - 0.05, 0.82):
            pick = best
            n_auto += 1
        elif use_llm and llm is not None:
            shelf = _shelf_context(df, idx, seen_video)
            li = _llm_pick(text, priced[:5], shelf, llm)
            if li is not None:
                pick = priced[li]
                n_llm += 1
        if pick is None:
            n_skip += 1
            continue
        if dry_run:
            print(f"  [dry] {idx}: {pick['name'][:50]} → {pick['barcode']} ({pick['score']:.2f})")
            continue
        df.at[idx, "barcode"] = pick["barcode"]
        if str(row.get("product_name", "") or "").strip().lower() in ("", "nan", "нет"):
            df.at[idx, "product_name"] = pick["name"]
        seen_video.append(f"{pick['barcode']} {pick['name'][:40]}")

    if not dry_run and (n_auto + n_llm):
        df.to_csv(final_path, index=False)
    print(f"  auto={n_auto} llm={n_llm} skip={n_skip}")
    return n_auto, n_llm, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet")
    ap.add_argument("--subset", default="all", choices=("beverage", "all"))
    ap.add_argument("--auto-threshold", type=float, default=0.90)
    ap.add_argument("--llm-threshold", type=float, default=0.72)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    resolve_file(
        args.final, args.ocr, args.catalog,
        subset=args.subset,
        auto_threshold=args.auto_threshold,
        llm_threshold=args.llm_threshold,
        use_llm=not args.no_llm,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
