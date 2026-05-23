"""Лучшее поле из K кадров трека."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "build_final_csv", ROOT / "ml" / "scripts" / "build_final_csv.py"
)
_bfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bfc)

VIDEOS = ("25_12-20", "26_12-20", "43_15")

FUSION_FIELDS = (
    "price_default", "price_card", "discount_amount", "id_sku",
    "print_datetime", "code", "additional_info", "special_symbols", "barcode",
)


def _row_text(r) -> str:
    parts = []
    for c in ("ocr_text", "paddle_text", "bottom_extra_text"):
        v = getattr(r, c, None) if not isinstance(r, dict) else r.get(c)
        if isinstance(v, str) and v.strip() and v != "nan":
            parts.append(v)
    return "\n".join(parts)


def _parse_row_fields(r) -> dict:
    text = _row_text(r)
    items_json = getattr(r, "ocr_items_json", None) if not isinstance(r, dict) else r.get("ocr_items_json")
    items = []
    if isinstance(items_json, str) and items_json.strip():
        try:
            items = json.loads(items_json)
        except Exception:
            pass
    prices = _bfc.parse_prices(items)
    bc_raw = getattr(r, "barcode_raw", "") if not isinstance(r, dict) else r.get("barcode_raw", "")
    qr_raw = getattr(r, "qr_raw", "") if not isinstance(r, dict) else r.get("qr_raw", "")
    qr = _bfc.parse_qr(qr_raw if isinstance(qr_raw, str) else "")
    barcode = _bfc.parse_barcode(text, bc_raw if isinstance(bc_raw, str) else "")
    if not barcode:
        qbc = qr.get("qr_code_barcode", "")
        if re.fullmatch(r"\d{12,13}", str(qbc).strip()):
            barcode = str(qbc).strip()
    return {
        "price_default": prices.get("price_default", ""),
        "price_card": prices.get("price_card", ""),
        "discount_amount": _bfc.parse_discount(text),
        "id_sku": _bfc.parse_id_sku(text),
        "print_datetime": _bfc.parse_print_datetime(text),
        "code": _bfc.parse_code(text),
        "additional_info": _bfc.parse_additional_info(text),
        "special_symbols": _bfc.parse_special_symbols(text, items),
        "barcode": barcode,
    }


def _vote_nonempty(values: list[str]) -> str:
    vals = [str(v).strip() for v in values if str(v or "").strip() and str(v).strip().lower() != "нет"]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def _best_sku(values: list[str]) -> str:
    cands = []
    for v in values:
        s = re.sub(r"\D", "", str(v))
        if len(s) == 12 and s[:3] in ("270", "370", "470", "170", "570", "770"):
            cands.append(s)
        elif len(s) >= 9:
            cands.append(s)
    if not cands:
        return _vote_nonempty(values)
    cands.sort(key=lambda x: (len(x) == 12, len(x)), reverse=True)
    return cands[0]


def _best_barcode(values: list[str]) -> str:
    return _bfc.choose_best_barcode(values)


def _best_datetime(values: list[str]) -> str:
    vals = [v for v in values if re.search(r"\d{1,2}\.\d{1,2}\.\d{2,4}", str(v))]
    return _vote_nonempty(vals) if vals else _vote_nonempty(values)


def _best_code(values: list[str]) -> str:
    vals = [
        v for v in values
        if _bfc.is_valid_layout_code(str(v))
    ]
    return _vote_nonempty(vals) if vals else ""


def fuse_track(sub: pd.DataFrame) -> dict:
    parsed = [_parse_row_fields(r) for _, r in sub.iterrows()]
    if not parsed:
        return {}
    return {
        "price_default": _vote_nonempty([p["price_default"] for p in parsed]),
        "price_card": _vote_nonempty([p["price_card"] for p in parsed]),
        "discount_amount": _vote_nonempty([p["discount_amount"] for p in parsed]),
        "id_sku": _best_sku([p["id_sku"] for p in parsed]),
        "print_datetime": _best_datetime([p["print_datetime"] for p in parsed]),
        "code": _best_code([p["code"] for p in parsed]),
        "additional_info": _vote_nonempty([p["additional_info"] for p in parsed]),
        "special_symbols": _vote_nonempty([p["special_symbols"] for p in parsed]),
        "barcode": _best_barcode([p["barcode"] for p in parsed]),
    }


def apply_fusion_to_final(video: str, dry_run: bool = False, in_path: Path | None = None) -> int:
    ocr_path = ROOT / "ml" / "output" / f"ocr_qr_{video}.csv"
    final_path = in_path or (ROOT / "ml" / "output" / f"final_eval_{video}.csv")
    if not ocr_path.is_file() or not final_path.is_file():
        print(f"  skip {video}: missing ocr/final")
        return 0
    ocr = pd.read_csv(ocr_path)
    final = pd.read_csv(final_path)
    if "track_id" not in ocr.columns:
        print(f"  skip {video}: no track_id")
        return 0

    track_fused: dict[int, dict] = {}
    for tid, sub in ocr.groupby("track_id"):
        if len(sub) < 2:
            continue
        track_fused[int(tid)] = fuse_track(sub)

    changed = 0
    for idx, row in final.iterrows():
        ts = float(row.get("frame_timestamp", 0))
        x0 = float(str(row.get("x_min", 0)).replace(",", "."))
        y0 = float(str(row.get("y_min", 0)).replace(",", "."))
        cand = ocr[
            (ocr["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
            & (ocr["x_min_orig"].astype(float).sub(x0).abs() < 12)
            & (ocr["y_min_orig"].astype(float).sub(y0).abs() < 12)
        ]
        if cand.empty:
            continue
        tid = int(cand.iloc[0]["track_id"])
        fused = track_fused.get(tid)
        if not fused:
            continue
        fill_if_empty = (
            "id_sku", "print_datetime", "code", "special_symbols",
            "discount_amount", "additional_info",
        )
        for field in fill_if_empty:
            if field not in final.columns:
                continue
            old_val = str(row.get(field, "") or "").strip()
            if old_val and old_val.lower() not in ("нет", "nan"):
                continue
            new_val = str(fused.get(field, "") or "").strip()
            if not new_val or new_val.lower() == "нет":
                continue
            final.at[idx, field] = new_val
            changed += 1

    if not dry_run and in_path is None:
        final.to_csv(final_path, index=False)
    elif not dry_run and in_path:
        final.to_csv(in_path, index=False)
        print(f"  wrote {in_path.name}")
    print(f"  {video}: {changed} field fills (empty only)")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy-dir", type=Path, default=None,
                    help="Писать в копии final_eval_<v>.csv в этой папке (не трогать ml/output)")
    args = ap.parse_args()
    videos = list(args.video) if args.video else (list(VIDEOS) if args.all else ["26_12-20"])
    total = 0
    for v in videos:
        in_path = None
        if args.copy_dir:
            args.copy_dir.mkdir(parents=True, exist_ok=True)
            src = ROOT / "ml" / "output" / f"final_eval_{v}.csv"
            dst = args.copy_dir / f"final_eval_{v}.csv"
            if not dst.exists() and src.is_file():
                import shutil
                shutil.copy2(src, dst)
            in_path = dst
        total += apply_fusion_to_final(v, dry_run=args.dry_run, in_path=in_path)
    print(f"Total updates: {total}")


if __name__ == "__main__":
    main()
