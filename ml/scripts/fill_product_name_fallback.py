"""Fallback product_name из OCR."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_STOP = {
    "цена", "цены", "карты", "карта", "карте", "карту",
    "акции", "акция", "акцион", "акционная",
    "руб", "руб.", "рубль", "руб/", "rubl",
    "штрих", "штрих-",
    "налог", "ндс",
    "сухое", "сухос", "белое", "красное", "розовое",
    "полусухое", "полусладкое", "сладкое", "игристое",
    "крепкое", "столовое",
    "лента", "ленты",
    "vino", "wine",
}

_TOK_RX = re.compile(r"[A-Za-zА-Яа-я0-9\-]{4,}")


def pick_tokens(text: str, max_n: int = 5) -> list[str]:
    if not isinstance(text, str):
        return []
    seen: set[str] = set()
    out: list[str] = []
    toks = sorted(set(_TOK_RX.findall(text)), key=lambda t: -len(t))
    for t in toks:
        low = t.lower()
        if low in _STOP or low in seen:
            continue
        if t.isdigit():
            continue
        if len(t) < 4:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--refresh-weak", action="store_true",
                    help="заменить слабые имена (нет/короткие/без brand-токенов)")
    args = ap.parse_args()

    final = pd.read_csv(args.final)
    ocr = pd.read_csv(args.ocr)

    rank0 = ocr[ocr.get("rank", 0) == 0] if "rank" in ocr.columns else ocr

    def collect_text(ts: int, xm: int, ym: int) -> str:
        def _join_sub(sub: pd.DataFrame) -> str:
            cols = ("top_extra_text", "paddle_text", "ocr_text", "bottom_extra_text")
            chunks: list[str] = []
            for col in cols:
                if col not in sub.columns:
                    continue
                for val in sub[col].dropna():
                    s = str(val).strip()
                    if s and s.lower() != "nan":
                        chunks.append(s)
            return "\n".join(chunks)

        cand = rank0[(rank0.frame_ts_ms == ts) & (rank0.x_min_orig == xm)
                     & (rank0.y_min_orig == ym)]
        if cand.empty:
            cand = ocr[ocr.frame_ts_ms == ts]
            if cand.empty:
                return ""
            return _join_sub(cand)
        tid = cand.iloc[0].track_id if "track_id" in cand.columns else None
        if tid is not None and "track_id" in ocr.columns:
            return _join_sub(ocr[ocr.track_id == tid])
        return _join_sub(cand)

    filled = 0
    new_names = []
    for r in final.itertuples(index=False):
        cur = getattr(r, "product_name", "")
        cur_s = str(cur).strip().lower() if cur is not None else ""
        text = collect_text(int(r.frame_timestamp), int(r.x_min), int(r.y_min))
        toks = pick_tokens(text)
        weak = (
            not cur_s or cur_s in ("нет", "nan")
            or (args.refresh_weak and len(cur_s) < 12)
            or (args.refresh_weak and toks and not any(t.lower() in cur_s for t in toks[:3]))
        )
        if not weak:
            new_names.append(cur)
            continue
        if len(toks) < 2:
            new_names.append(cur if cur_s and cur_s not in ("нет", "nan") else "")
            continue
        joined = " ".join(toks)
        letters = sum(c.isalpha() for c in joined)
        if len(joined) < 10 or letters / max(len(joined), 1) < 0.5:
            new_names.append(cur if cur_s and cur_s not in ("нет", "nan") else "")
            continue
        new_names.append(joined)
        filled += 1

    final["product_name"] = new_names
    final.to_csv(args.final, index=False, encoding="utf-8")
    print(f"Fallback заполнил: {filled}/{len(final)} → {args.final}")


if __name__ == "__main__":
    main()
