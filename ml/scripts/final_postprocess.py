"""Постобработка final CSV."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

NET_IF_EMPTY = [
    "print_datetime", "id_sku", "code", "special_symbols",
    "price_discount",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

_DRY_RX = re.compile(r"(?<![\w/])кр\.?\s*сух", re.IGNORECASE)
_SEMI_RX = re.compile(r"п\s*/?\s*сух|полусух", re.IGNORECASE)
_CATALOG_NAME_RX = re.compile(
    r"^(Вино|Мед|Виски|Коньяк|Водка|Пиво|Сок|Чай|Кофе|Сыр|Масло)\b",
    re.IGNORECASE,
)
_COUNTRY_RX = re.compile(r"\([A-Za-zА-Яа-я .\-]{3,30}\)\s*(?:\d|$|\.)")
_LATIN_OK = frozenset(
    "doc dop igt cuvee brut reserve reserva rosso bianco spumante cl DOC DOP IGT".split()
)


def _is_empty_or_no(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no", "n/a")


def _as_price(x):
    try:
        val = float(str(x).strip().replace(",", "."))
    except Exception:
        return None
    if val != val or val <= 0:
        return None
    return val


def _infer_dry_from_text(text: str) -> str:
    low = str(text or "").lower()
    if _SEMI_RX.search(low):
        return ""
    if _DRY_RX.search(low):
        return "Сухое"
    return ""


def _map_ocr_track_text(final_row, ocr_df) -> str:
    """OCR всех K кадров трека (ocr + paddle + bottom)."""
    if ocr_df is None or ocr_df.empty or "track_id" not in ocr_df.columns:
        return _map_ocr_row(final_row, ocr_df)
    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(str(final_row.get("x_min", 0)).replace(",", "."))
    y0 = float(str(final_row.get("y_min", 0)).replace(",", "."))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
    ]
    if cand.empty:
        return _map_ocr_row(final_row, ocr_df)
    tid = int(cand.iloc[0]["track_id"])
    sub = ocr_df[ocr_df["track_id"] == tid]
    chunks: list[str] = []
    for _, r in sub.iterrows():
        for c in ("ocr_text", "paddle_text", "bottom_extra_text"):
            v = str(r.get(c, "") or "")
            if v.strip() and v != "nan":
                chunks.append(v)
    return "\n".join(chunks)


def _track_fused_fields(final_row, ocr_df) -> dict:
    if ocr_df is None or ocr_df.empty or "track_id" not in ocr_df.columns:
        return {}
    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(str(final_row.get("x_min", 0)).replace(",", "."))
    y0 = float(str(final_row.get("y_min", 0)).replace(",", "."))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
    ]
    if cand.empty:
        return {}
    tid = int(cand.iloc[0]["track_id"])
    sub = ocr_df[ocr_df["track_id"] == tid]
    if sub.empty:
        return {}
    try:
        from multi_frame_field_fusion import fuse_track
        return fuse_track(sub)
    except Exception:
        return {}


FUSION_FILL_FIELDS = (
    "id_sku", "print_datetime", "code", "special_symbols",
    "additional_info", "discount_amount",
)


def _needs_reparse(field: str, val) -> bool:
    if _is_empty_or_no(val):
        return True
    if field == "code":
        try:
            from build_final_csv import is_valid_layout_code
            return not is_valid_layout_code(str(val))
        except Exception:
            return True
    return False


def _reparse_track_fields(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    try:
        from build_final_csv import (
            is_valid_layout_code,
            parse_code,
            parse_id_sku,
            parse_print_datetime,
        )
    except Exception:
        return {}
    out: dict[str, str] = {}
    sku = parse_id_sku(text)
    if sku:
        out["id_sku"] = sku
    code = parse_code(text)
    if code and is_valid_layout_code(code):
        out["code"] = code
    dt = parse_print_datetime(text)
    if dt:
        out["print_datetime"] = dt
    return out


def _map_ocr_row(final_row, ocr_df) -> str:
    """Склеить OCR-текст по bbox+timestamp для additional_info fallback."""
    if ocr_df is None or ocr_df.empty:
        return ""
    ts = float(final_row.get("frame_timestamp", 0))
    x0 = float(final_row.get("x_min", 0))
    y0 = float(final_row.get("y_min", 0))
    cand = ocr_df[
        (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 10)
        & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 8)
        & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 8)
    ]
    if cand.empty:
        return ""
    r = cand.iloc[0]
    parts = [str(r.get(c, "") or "") for c in ("ocr_text", "paddle_text", "bottom_extra_text")]
    return "\n".join(p for p in parts if p and p != "nan")


def _ean_checksum_ok(code: str) -> bool:
    if not re.fullmatch(r"\d{13}", code):
        return False
    digits = [int(c) for c in code]
    check = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return check == digits[-1]


def _norm_bc(x) -> str:
    s = str(x or "").strip()
    if s.lower() in ("", "nan"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    if "e" in s.lower():
        try:
            s = f"{int(float(s))}"
        except Exception:
            pass
    s = re.sub(r"\D", "", s)
    return s


def _has_meaningful_price(x) -> bool:
    s = str(x or "").strip()
    if s.lower() in ("", "нет", "nan"):
        return False
    try:
        return float(s.replace(",", ".")) > 0
    except Exception:
        return False


def _tag_confidence(row) -> int:
    score = 0
    bc = _norm_bc(row.get("barcode"))
    if len(bc) == 13 and _ean_checksum_ok(bc):
        score += 50
    elif len(bc) >= 12:
        score += 18
    if _has_meaningful_price(row.get("price_default")) or _has_meaningful_price(row.get("price_card")):
        score += 30
    qr = _norm_bc(row.get("qr_code_barcode"))
    if len(qr) >= 12:
        score += 22
    if str(row.get("id_sku") or "").strip():
        score += 8
    pn = str(row.get("product_name") or "").strip()
    if len(pn) >= 15 and sum(c.isalpha() for c in pn) / max(len(pn), 1) > 0.55:
        score += 12
    return score


def _bbox_overlap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    sm = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / sm if sm > 0 else 0.0


def _token_mixed_script(token: str) -> bool:
    has_lat = bool(re.search(r"[A-Za-z]", token))
    has_cyr = bool(re.search(r"[а-яё]", token, re.IGNORECASE))
    return has_lat and has_cyr


def _has_trusted_barcode(row) -> bool:
    """Имя из каталога доверяем только при валидном EAN-13 (не partial 12-digit OCR)."""
    bc = _norm_bc(row.get("barcode"))
    return len(bc) == 13 and _ean_checksum_ok(bc)


def _is_weak_product_name(name: str, *, has_valid_bc: bool) -> bool:
    s = str(name or "").strip()
    if s.lower() in ("", "нет", "nan") or s.lower().startswith("nan "):
        return True
    if re.fullmatch(r"[\d\-+./\\]+", s):
        return True
    if re.match(r"^[\d\-+./\\]", s) and len(s) < 12:
        return True
    if len(s) < 10:
        return True
    if re.search(r"[\[\]{}]", s):
        return True
    if has_valid_bc:
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / max(len(s), 1) < 0.45:
        return True
    toks = s.split()
    if len(toks) <= 2 and all(len(t) < 8 for t in toks):
        return True
    if any(_token_mixed_script(t) for t in toks):
        return True
    latin_words = [t for t in toks if re.fullmatch(r"[A-Za-z0-9.\-/]{2,}", t)]
    cyr_words = [t for t in toks if re.search(r"[а-яё]{3,}", t, re.IGNORECASE)]
    if latin_words and cyr_words:
        unknown = [t for t in latin_words if t.lower() not in _LATIN_OK and len(t) >= 4]
        if unknown:
            return True
    if _COUNTRY_RX.search(s):
        return False
    if _CATALOG_NAME_RX.search(s):
        return False
    # Осмысленное кириллическое название (в т.ч. от LLM)
    if len(s) >= 10 and letters / len(s) > 0.50:
        if sum(bool(re.search(r"[а-яё]", c, re.I)) for c in s) >= 5:
            return False
    return True


def _finalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """После stabilize: пустые/NaN product_name → «нет», повторная очистка имён."""
    out = df.copy()
    if "product_name" in out.columns:
        out["product_name"] = out["product_name"].astype(object).where(out["product_name"].notna(), "")
    for idx, row in out.iterrows():
        bc_ok = _has_trusted_barcode(row)
        pn = str(row.get("product_name") or "").strip()
        if _is_weak_product_name(pn, has_valid_bc=bc_ok):
            out.at[idx, "product_name"] = "нет"
        elif _is_empty_or_no(pn):
            out.at[idx, "product_name"] = "нет"
    return out


def _row_has_signal(row) -> bool:
    if _has_meaningful_price(row.get("price_default")) or _has_meaningful_price(row.get("price_card")):
        return True
    if len(_norm_bc(row.get("barcode"))) >= 12:
        return True
    if len(_norm_bc(row.get("qr_code_barcode"))) >= 12:
        return True
    return False


def _load_barcode_index(catalog_path: Path | None) -> dict[str, int]:
    if catalog_path is None or not catalog_path.is_file():
        return {}
    try:
        from match_to_catalog import _build_index, _coerce_catalog_columns

        if catalog_path.suffix.lower() == ".parquet":
            cat = pd.read_parquet(catalog_path)
        else:
            cat = pd.read_csv(catalog_path, sep=";")
        _, bc_idx = _build_index(_coerce_catalog_columns(cat))
        print(f"  catalog barcode index: {len(bc_idx)} EAN")
        return bc_idx
    except Exception as exc:
        print(f"  catalog index skip: {exc}")
        return {}


def _barcode_in_catalog(bc: str, bc_idx: dict[str, int]) -> bool:
    if not bc or len(bc) < 12 or not bc_idx:
        return len(bc) >= 12 and not bc_idx
    if bc in bc_idx:
        return True
    if len(bc) == 13 and bc[:12] in bc_idx:
        return True
    return False


def _row_rank(row) -> tuple:
    bc = _norm_bc(row.get("barcode"))
    area = (int(row["x_max"]) - int(row["x_min"])) * (int(row["y_max"]) - int(row["y_min"]))
    return (1 if len(bc) >= 12 else 0, _tag_confidence(row), area)


def _sanitize_fields(row: dict, *, bc_idx: dict[str, int] | None = None) -> None:
    bc = _norm_bc(row.get("barcode"))
    if bc_idx and len(bc) >= 12 and not _barcode_in_catalog(bc, bc_idx):
        row["barcode"] = ""
        bc = ""
    if len(bc) == 13 and not _ean_checksum_ok(bc):
        row["barcode"] = ""
        bc = ""
    elif len(bc) in (12, 13):
        row["barcode"] = bc
    elif bc:
        row["barcode"] = ""

    has_valid_bc = _has_trusted_barcode(row)
    pn = str(row.get("product_name") or "").strip()
    if _is_weak_product_name(pn, has_valid_bc=has_valid_bc):
        row["product_name"] = ""


def stabilize_dataframe(
    df: pd.DataFrame,
    *,
    min_confidence: int = 15,
    dedup_ms: int = 2000,
    dedup_iou: float = 0.5,
    bc_idx: dict[str, int] | None = None,
) -> pd.DataFrame:
    rows = [dict(r) for r in df.to_dict("records")]
    for r in rows:
        _sanitize_fields(r, bc_idx=bc_idx)

    n0 = len(rows)
    rows = [r for r in rows if _row_has_signal(r)]
    n1 = len(rows)
    if min_confidence > 0:
        rows = [r for r in rows if _tag_confidence(r) >= min_confidence]
    n2 = len(rows)

    rows.sort(key=lambda r: _row_rank(r), reverse=True)
    kept: list[dict] = rows if dedup_ms <= 0 else []
    if dedup_ms > 0:
        for r in rows:
            rb = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
            rts = int(r["frame_timestamp"])
            merged = False
            for i, k in enumerate(kept):
                kb = (int(k["x_min"]), int(k["y_min"]), int(k["x_max"]), int(k["y_max"]))
                if abs(rts - int(k["frame_timestamp"])) <= dedup_ms and _bbox_overlap(rb, kb) >= dedup_iou:
                    if _row_rank(r) > _row_rank(k):
                        kept[i] = r
                    merged = True
                    break
            if not merged:
                kept.append(r)

    out = pd.DataFrame(kept) if kept else df.iloc[0:0].copy()
    print(f"  stabilize: {n0} → signal {n1} → conf≥{min_confidence} {n2} → dedup {len(out)}")
    return out


def postprocess(
    df: pd.DataFrame,
    ocr_df: pd.DataFrame | None = None,
    *,
    stabilize: bool = True,
    min_confidence: int = 15,
    dedup_ms: int = 2000,
    bc_idx: dict[str, int] | None = None,
    finalize_names: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    for col in NET_IF_EMPTY:
        if col in out.columns:
            out[col] = out[col].astype(object).where(out[col].notna(), "")
    if "discount_amount" in out.columns:
        out["discount_amount"] = out["discount_amount"].astype(object).where(
            out["discount_amount"].notna(), "")
    if "additional_info" in out.columns:
        out["additional_info"] = out["additional_info"].astype(object).where(
            out["additional_info"].notna(), "")
    for idx, row in out.iterrows():
        bc = re.sub(r"\D", "", str(row.get("barcode") or ""))
        p1 = _as_price(row.get("price_default"))
        p4 = _as_price(row.get("price_card"))
        if len(bc) >= 12 and p1 is not None and p4 is not None and p4 < p1 * 0.98:
            pct = int((p1 - p4) / p1 * 100)
            if 5 <= pct <= 90:
                out.at[idx, "discount_amount"] = f"-{pct}%"

        if _is_empty_or_no(row.get("additional_info")):
            ocr_text = _map_ocr_track_text(row, ocr_df)
            hint = _infer_dry_from_text(ocr_text)
            if hint:
                out.at[idx, "additional_info"] = hint

        fused = _track_fused_fields(row, ocr_df)
        for field in FUSION_FILL_FIELDS:
            if field not in out.columns:
                continue
            if not _is_empty_or_no(out.at[idx, field]):
                continue
            new_val = str(fused.get(field, "") or "").strip()
            if new_val and not _is_empty_or_no(new_val):
                out.at[idx, field] = new_val

        track_text = _map_ocr_track_text(row, ocr_df)
        reparsed = _reparse_track_fields(track_text)
        for field in ("id_sku", "code", "print_datetime"):
            if field not in out.columns:
                continue
            if not _needs_reparse(field, out.at[idx, field]):
                continue
            new_val = str(reparsed.get(field, "") or "").strip()
            if new_val and not _is_empty_or_no(new_val):
                out.at[idx, field] = new_val
            elif field == "code":
                out.at[idx, field] = ""

        for col in NET_IF_EMPTY:
            if col not in out.columns:
                continue
            if _is_empty_or_no(out.at[idx, col]):
                out.at[idx, col] = "нет"

    if finalize_names and not stabilize:
        return _finalize_text_fields(out)

    if stabilize:
        out = stabilize_dataframe(
            out,
            min_confidence=min_confidence,
            dedup_ms=dedup_ms,
            bc_idx=bc_idx,
        )
        out = _finalize_text_fields(out)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None, help="final CSV path")
    ap.add_argument("--all", action="store_true", help="все final_eval_*.csv в ml/output")
    ap.add_argument("--dir", type=Path, default=None, help="директория с final/ocr CSV")
    ap.add_argument("--no-stabilize", action="store_true",
                    help="не фильтровать слабые строки и не дедуплицировать")
    ap.add_argument("--min-confidence", type=int, default=15)
    ap.add_argument("--dedup-ms", type=int, default=2000,
                    help="окно spatial dedup, мс (0=только фильтр слабых строк)")
    ap.add_argument("--catalog", type=Path, default=None,
                    help="каталог для проверки barcode (parquet/csv)")
    ap.add_argument("--finalize-names", action="store_true",
                    help="только очистка product_name, без stabilize")
    args = ap.parse_args()

    base = args.dir or (ROOT / "ml" / "output")
    bc_idx = _load_barcode_index(args.catalog)
    paths: list[Path]
    if args.all:
        paths = sorted(p for p in base.glob("final_eval_*.csv")
                       if "_fused" not in p.stem)
    elif args.final:
        paths = [Path(args.final)]
    else:
        ap.error("укажите --final или --all")

    for p in paths:
        stem = p.stem.replace("final_eval_", "")
        ocr_path = base / f"ocr_qr_{stem}.csv"
        ocr_df = pd.read_csv(ocr_path) if ocr_path.exists() else None
        df = pd.read_csv(p)
        out = postprocess(
            df,
            ocr_df,
            stabilize=not args.no_stabilize and not args.finalize_names,
            min_confidence=args.min_confidence,
            dedup_ms=args.dedup_ms,
            bc_idx=bc_idx or None,
            finalize_names=args.finalize_names,
        )
        out.to_csv(p, index=False, encoding="utf-8")
        print(f"postprocessed {p.name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
