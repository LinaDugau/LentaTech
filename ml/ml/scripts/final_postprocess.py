"""Безопасная постобработка final_*.csv для near-threshold пар."""
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
    if len(sub) < 2:
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


def postprocess(df: pd.DataFrame, ocr_df: pd.DataFrame | None = None) -> pd.DataFrame:
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

        for col in NET_IF_EMPTY:
            if col not in out.columns:
                continue
            if _is_empty_or_no(out.at[idx, col]):
                out.at[idx, col] = "нет"

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None, help="final CSV path")
    ap.add_argument("--all", action="store_true", help="все final_eval_*.csv в ml/output")
    args = ap.parse_args()

    paths: list[Path]
    if args.all:
        paths = sorted(p for p in (ROOT / "ml" / "output").glob("final_eval_*.csv")
                       if "_fused" not in p.stem)
    elif args.final:
        paths = [Path(args.final)]
    else:
        ap.error("укажите --final или --all")

    for p in paths:
        stem = p.stem.replace("final_eval_", "")
        ocr_path = ROOT / "ml" / "output" / f"ocr_qr_{stem}.csv"
        ocr_df = pd.read_csv(ocr_path) if ocr_path.exists() else None
        df = pd.read_csv(p)
        out = postprocess(df, ocr_df)
        out.to_csv(p, index=False, encoding="utf-8")
        print(f"postprocessed {p.name} ({len(out)} rows)")


if __name__ == "__main__":
    main()
