"""product_name через Qwen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
_GGUF_CANDIDATES = (
    ROOT / "ml" / "weights" / "qwen2.5-3b-q4_k_m.gguf",
    ROOT / "weights" / "qwen2.5-3b-q4_k_m.gguf",
)
DEFAULT_GGUF = next((p for p in _GGUF_CANDIDATES if p.is_file()), _GGUF_CANDIDATES[0])

_LLM = None


def get_llm(gguf_path: Path | None = None,
            n_ctx: int = 2048, n_gpu_layers: int = -1):
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        p = Path(gguf_path) if gguf_path else DEFAULT_GGUF
        if not p.exists():
            raise FileNotFoundError(
                f"GGUF не найден: {p}\n"
                "Скачайте Qwen2.5-3B-Instruct Q4_K_M или положите файл сюда.")
        _LLM = Llama(model_path=str(p),
                     n_ctx=n_ctx,
                     n_gpu_layers=n_gpu_layers,  # -1 = всё на GPU (Metal/CUDA)
                     verbose=False)
    return _LLM

SYSTEM = (
    "Ты помощник, который извлекает названия товаров из текста OCR с ценника "
    "магазина Лента. Текст шумный (опечатки OCR, лишние цифры цены, бренды). "
    "Извлеки ТОЛЬКО название товара (бренд, тип, объём/вес), без цен/штрихкодов/дат. "
    "Если в тексте нет наименования товара — ответь ровно: нет\n"
    "Не объясняй. Только название (одна строка) или «нет»."
)

EXAMPLES = [
    ("Напиток SANTO STEFANO Rosso Россия 0,25L 252,63 129,99",
     "Напиток SANTO STEFANO Rosso 0,25L"),
    ("МЕД ПОТАПЫЧ Натуральный липовый банка 500г 415,79 316,99",
     "Мед ПОТАПЫЧ натуральный липовый 500г"),
    ("99 ппшшмп 30% 1234567",
     "нет"),
]


def build_messages(ocr_text: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    for src, ans in EXAMPLES:
        msgs.append({"role": "user", "content": f"Текст: {src}"})
        msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": f"Текст: {ocr_text}"})
    return msgs


def extract_one(ocr_text: str, max_tokens: int = 80) -> str:
    if not ocr_text or len(ocr_text.strip()) < 5:
        return ""
    text = ocr_text.replace("\n", " ").strip()
    if len(text) > 600:
        text = text[:600]
    try:
        out = get_llm().create_chat_completion(
            messages=build_messages(text),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        ans = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return ""
    ans = ans.split("\n")[0].strip().strip('"').strip("«»")
    if ans.lower() in ("нет", "no", "none", "n/a", ""):
        return "нет"
    if len(ans) < 3 or len(ans) > 200:
        return ""
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True,
                    help="ml/output/final_<video>.csv")
    ap.add_argument("--ocr", required=True,
                    help="ml/output/ocr_qr_<video>.csv (источник текста)")
    ap.add_argument("--gguf", default=None,
                    help=f"путь к GGUF (по умолчанию {DEFAULT_GGUF})")
    ap.add_argument("--only-empty", action="store_true",
                    help="заполнять только строки с пустым/«нет» product_name")
    ap.add_argument("--min-confidence", type=int, default=0,
                    help="LLM только для строк с ценой (confidence proxy ≥ N)")
    args = ap.parse_args()
    if args.gguf:
        get_llm(Path(args.gguf))
    else:
        get_llm()  # ленивая инициализация на DEFAULT_GGUF

    final = pd.read_csv(args.final)
    ocr = pd.read_csv(args.ocr)

    rank0 = ocr[ocr.get("rank", 0) == 0] if "rank" in ocr.columns else ocr

    def match_text(ts: int, x_min: int, y_min: int) -> str:
        def _collect(sub: pd.DataFrame) -> str:
            chunks: list[str] = []
            for col in ("top_extra_text", "paddle_text", "ocr_text", "bottom_extra_text"):
                if col not in sub.columns:
                    continue
                for val in sub[col].dropna():
                    s = str(val).strip()
                    if s and s.lower() != "nan":
                        chunks.append(s)
            return "\n".join(chunks)

        cand = rank0[(rank0.frame_ts_ms == ts)
                     & (rank0.x_min_orig == x_min)
                     & (rank0.y_min_orig == y_min)]
        if cand.empty:
            cand = ocr[ocr.frame_ts_ms == ts]
            if cand.empty:
                return ""
            return _collect(cand)
        tid = cand.iloc[0].track_id if "track_id" in cand.columns else None
        if tid is not None and "track_id" in ocr.columns:
            return _collect(ocr[ocr.track_id == tid])
        return _collect(cand)

    names = []
    for r in tqdm(final.itertuples(index=False), total=len(final),
                  desc=Path(args.final).stem):
        cur = getattr(r, "product_name", "")
        cur_s = str(cur or "").strip().lower()
        if args.only_empty and cur_s not in ("", "нет", "nan"):
            names.append(cur)
            continue
        if args.min_confidence > 0:
            has_price = False
            for fld in ("price_default", "price_card"):
                v = getattr(r, fld, "")
                try:
                    if float(str(v).replace(",", ".")) > 0:
                        has_price = True
                        break
                except Exception:
                    pass
            if not has_price:
                names.append(cur if cur_s not in ("", "nan") else "нет")
                continue
        text = match_text(int(r.frame_timestamp), int(r.x_min), int(r.y_min))
        name = extract_one(text) if text else ""
        if not name or str(name).strip().lower() in ("", "нет", "nan"):
            names.append("нет")
        else:
            names.append(name)

    final["product_name"] = names
    final.to_csv(args.final, index=False, encoding="utf-8")
    n_named = sum(1 for n in names if n and n != "нет")
    print(f"product_name заполнено: {n_named}/{len(names)}")


if __name__ == "__main__":
    main()
