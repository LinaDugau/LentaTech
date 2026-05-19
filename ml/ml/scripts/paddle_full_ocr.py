"""PaddleOCR на всех hires-crops + merge в ocr_qr_*.csv."""
from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

_OCR = None


def _limit_threads_for_paddle() -> None:
    """Paddle + OpenBLAS часто падают с SIGSEGV при OMP>1 в одном процессе с PyTorch."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def get_paddle():
    global _OCR
    if _OCR is None:
        _limit_threads_for_paddle()
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="ru",
            device="cpu",
        )
    return _OCR


def paddle_text_for_image(img_path: Path, min_score: float = 0.3) -> tuple[str, list]:
    """Возвращает (склеенный текст, items=(text,score,height))."""
    img = cv2.imread(str(img_path))
    if img is None:
        return "", []
    try:
        result = get_paddle().predict(img)
    except Exception:
        return "", []
    if not result:
        return "", []
    res = result[0].json.get("res", {}) if hasattr(result[0], "json") else {}
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    polys = res.get("rec_polys", [])

    items = []
    kept_texts = []
    for i, (t, s) in enumerate(zip(texts, scores)):
        if s < min_score:
            continue
        h = 0
        if i < len(polys):
            try:
                ys = [p[1] for p in polys[i]]
                h = max(ys) - min(ys)
            except Exception:
                h = 0
        items.append((t, float(s), float(h)))
        kept_texts.append(t)
    return "\n".join(kept_texts), items


def process(
    ocr_csv: Path,
    progress_hook: Optional[Callable[[int, int], None]] = None,
):
    df = pd.read_csv(ocr_csv)
    if "hires_crop" not in df.columns:
        print(f"  {ocr_csv.name}: no hires_crop column — skip")
        return
    if "paddle_text" not in df.columns:
        df["paddle_text"] = ""

    n_new = 0
    n_with_text = 0
    n_total = len(df)
    for i, (idx, row) in enumerate(
        tqdm(df.iterrows(), total=n_total, desc=ocr_csv.stem)
    ):
        if progress_hook and (i % 8 == 0 or i == n_total - 1):
            progress_hook(i + 1, n_total)
        existing = row.get("paddle_text")
        if isinstance(existing, str) and existing.strip():
            n_with_text += 1
            continue
        cp = row.get("hires_crop")
        if not isinstance(cp, str):
            continue
        p = ROOT / cp
        if not p.exists():
            continue
        text, _items = paddle_text_for_image(p)
        if text:
            df.at[idx, "paddle_text"] = text
            n_with_text += 1
        n_new += 1
    if progress_hook and n_total > 0:
        progress_hook(n_total, n_total)

    def _merge(row):
        base = str(row.get("ocr_text", "") or "").strip()
        pt = str(row.get("paddle_text", "") or "").strip()
        if not pt:
            return base
        if pt and base:
            return base + "\n" + pt
        return pt or base

    df["ocr_text"] = df.apply(_merge, axis=1)

    df.to_csv(ocr_csv, index=False)
    print(f"  {ocr_csv.name}: processed new={n_new}, paddle non-empty={n_with_text}/{len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ocr-csv",
        type=Path,
        default=None,
        help="один ocr_qr_*.csv (subprocess из pipeline, изолированный OMP для Paddle)",
    )
    ap.add_argument("--videos", nargs="*", default=None,
                    help="например 26_12-20 (по умолчанию все)")
    args = ap.parse_args()

    if args.ocr_csv is not None:
        p = args.ocr_csv.resolve()
        if not p.exists():
            print(f"  --ocr-csv not found: {p}")
            raise SystemExit(1)
        print(f"\n=== {p.name} ===")
        process(p)
        return

    files = sorted((ROOT / "ml" / "output").glob("ocr_qr_*.csv"))
    files = [f for f in files if "_2-10" not in f.name]  # skip unlabeled
    if args.videos:
        wanted = set(args.videos)
        files = [f for f in files if any(v in f.name for v in wanted)]

    for f in files:
        print(f"\n=== {f.name} ===")
        process(f)


if __name__ == "__main__":
    main()
