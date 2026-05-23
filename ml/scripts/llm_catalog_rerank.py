"""LLM rerank кандидатов каталога."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
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
from llm_product_name import get_llm  # noqa: E402

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _empty_barcode(x) -> bool:
    s = re.sub(r"\D", "", str(x or ""))
    return len(s) < 12


def _empty_name(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no")


def _load_name_index(catalog_path: Path) -> tuple[list[str], list[str], list[str]]:
    """names_norm, barcodes, raw names."""
    if catalog_path.suffix.lower() == ".parquet":
        cat = pd.read_parquet(catalog_path)
    else:
        cat = pd.read_csv(catalog_path, sep=";", encoding="cp1251", on_bad_lines="skip")
    ren = {}
    for c in cat.columns:
        k = str(c).strip().lower()
        if k in ("fullname", "full_name", "product_name", "name"):
            ren[c] = "name"
        elif k in ("code", "barcode", "ean"):
            ren[c] = "barcode"
    cat = cat.rename(columns=ren)
    names: list[str] = []
    barcodes: list[str] = []
    raw: list[str] = []
    for _, row in cat.iterrows():
        name = str(row.get("name", "") or "").strip()
        bc = re.sub(r"\D", "", str(row.get("barcode", "") or ""))
        if len(name) < 4 or len(bc) < 12:
            continue
        names.append(_mtc._normalize(name))
        barcodes.append(bc[:13])
        raw.append(name)
    return names, barcodes, raw


def _top_candidates(
    ocr_text: str,
    names_norm: list[str],
    barcodes: list[str],
    raw_names: list[str],
    *,
    limit: int = 5,
    min_score: float = 72.0,
) -> list[dict]:
    if not ocr_text.strip() or not HAS_RAPIDFUZZ:
        return []
    query = _mtc._normalize(ocr_text)
    if len(query) < 8:
        return []
    hits = rf_process.extract(
        query, names_norm, scorer=fuzz.token_set_ratio, limit=limit * 3,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for name_norm, score, idx in hits:
        if score < min_score:
            continue
        bc = barcodes[idx]
        if bc in seen:
            continue
        seen.add(bc)
        out.append({"barcode": bc, "name": raw_names[idx], "score": float(score)})
        if len(out) >= limit:
            break
    return out


def _shelf_context(df: pd.DataFrame, idx: int, max_items: int = 8) -> str:
    rows = []
    ts = float(df.at[idx, "frame_timestamp"])
    for j, row in df.iterrows():
        if j == idx:
            continue
        if abs(float(row.get("frame_timestamp", 0)) - ts) > 8000:
            continue
        name = str(row.get("product_name", "") or "").strip()
        bc = re.sub(r"\D", "", str(row.get("barcode", "") or ""))
        if name and name.lower() not in ("нет", "nan"):
            rows.append(f"- {name}" + (f" (EAN {bc})" if len(bc) >= 12 else ""))
        if len(rows) >= max_items:
            break
    return "\n".join(rows) if rows else "нет данных"


def _llm_pick(ocr_text: str, candidates: list[dict], shelf: str) -> dict | None:
    if not candidates:
        return None
    opts = "\n".join(
        f"{i}. EAN={c['barcode']} | {c['name'][:120]} (score={c['score']:.0f})"
        for i, c in enumerate(candidates)
    )
    prompt = (
        "Ты сопоставляешь шумный OCR-текст ценника с каталогом товаров Лента.\n"
        "Выбери ОДИН наиболее подходящий вариант из списка или ответь NONE.\n"
        "Не придумывай EAN — только номер из списка (0..N-1) или NONE.\n"
        "Учитывай OCR-ошибки, бренд, объём, тип напитка.\n\n"
        f"OCR:\n{ocr_text[:800]}\n\n"
        f"Уже распознано рядом на полке:\n{shelf or '—'}\n\n"
        f"Кандидаты:\n{opts}\n\n"
        "Ответ: одно число (индекс) или NONE."
    )
    try:
        out = get_llm().create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
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
    if 0 <= pick < len(candidates):
        return candidates[pick]
    return None


def rerank_file(
    final_path: Path,
    ocr_path: Path,
    catalog_path: Path,
    *,
    min_fuzz: float = 78.0,
    fill_barcode: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    if not final_path.is_file() or not ocr_path.is_file():
        return 0, 0
    df = pd.read_csv(final_path, dtype={"barcode": str}, keep_default_na=False)
    ocr = pd.read_csv(ocr_path)
    names_norm, barcodes, raw_names = _load_name_index(catalog_path)
    print(f"  index: {len(names_norm)} SKU from {catalog_path.name}")

    n_cand = 0
    n_changed = 0
    for idx, row in tqdm(list(df.iterrows()), desc=final_path.stem):
        if not _empty_barcode(row.get("barcode")):
            continue
        text = _foc._track_text(row, ocr)
        if len(text.strip()) < 12:
            continue
        cands = _top_candidates(
            text, names_norm, barcodes, raw_names, min_score=min_fuzz,
        )
        if not cands:
            continue
        n_cand += 1
        shelf = _shelf_context(df, idx)
        pick = _llm_pick(text, cands, shelf)
        if pick is None:
            continue
        bc = pick["barcode"]
        if len(bc) != 13 or not _ean_checksum_ok(bc):
            continue
        if dry_run:
            print(f"  [dry] row {idx}: {pick['name'][:60]} → {bc}")
            n_changed += 1
            continue
        if _empty_name(row.get("product_name")):
            df.at[idx, "product_name"] = pick["name"]
        if fill_barcode and _empty_barcode(row.get("barcode")):
            df.at[idx, "barcode"] = bc
        n_changed += 1

    if not dry_run and n_changed:
        df.to_csv(final_path, index=False)
    print(f"  candidates={n_cand}, changed={n_changed}")
    return n_cand, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet")
    ap.add_argument("--min-fuzz", type=float, default=78.0)
    ap.add_argument("--fill-barcode", action="store_true", help="записать EAN если LLM выбрал (осторожно)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rerank_file(
        args.final, args.ocr, args.catalog,
        min_fuzz=args.min_fuzz,
        fill_barcode=args.fill_barcode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
