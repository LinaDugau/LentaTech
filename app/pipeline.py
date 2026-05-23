"""Обёртка над ml.pipeline.PriceTagPipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib"

from app.config import (
    BBOX_COLOR,
    BBOX_THICKNESS,
    CATALOG_PATH,
    DETECTOR_WEIGHTS_PATH,
    LLM_WEIGHTS_PATH,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    TEXT_COLOR,
    TEXT_THICKNESS,
)

logger = logging.getLogger(__name__)

_PIPELINE: Any = None


def _ensure_ml_package_path() -> None:
    nested_project_root = Path("/app/ml")
    if (nested_project_root / "ml" / "pipeline.py").is_file():
        nested_project_root_str = str(nested_project_root)
        if nested_project_root_str not in sys.path:
            sys.path.insert(0, nested_project_root_str)


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _ensure_ml_package_path()
        from ml.pipeline import PriceTagPipeline

        use_llm = os.getenv("USE_LLM", "1") not in ("0", "false", "False")
        use_paddle = os.getenv("USE_PADDLE_OCR", "1") not in ("0", "false", "False")
        use_aggressive = os.getenv("USE_AGGRESSIVE_DECODE", "1") not in ("0", "false", "False")
        use_submission_finalize = os.getenv("USE_SUBMISSION_FINALIZE", "1") not in (
            "0", "false", "False")
        frame_step = int(os.getenv("ML_FRAME_STEP", "3"))
        top_k = int(os.getenv("ML_TOP_K", "3"))
        tta_env = os.getenv("ML_USE_TTA", "").strip()
        use_tta = tta_env not in ("0", "false", "False") if tta_env else use_submission_finalize
        logger.info(
            "Loading PriceTagPipeline "
            f"(use_llm={use_llm}, submission_finalize={use_submission_finalize}, "
            f"tta={use_tta}, frame_step={frame_step}, "
            f"weights={DETECTOR_WEIGHTS_PATH}, catalog={CATALOG_PATH})..."
        )
        _PIPELINE = PriceTagPipeline(
            weights=DETECTOR_WEIGHTS_PATH,
            frame_step=frame_step,
            top_k=top_k,
            use_llm=use_llm,
            gguf_path=LLM_WEIGHTS_PATH,
            use_paddle=use_paddle,
            use_aggressive_decode=use_aggressive,
            use_submission_finalize=use_submission_finalize,
            use_tta=use_tta,
            catalog_path=CATALOG_PATH,
        )
        logger.info("PriceTagPipeline loaded")
    return _PIPELINE


def process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Тот же пайплайн, что python ml/pipeline.py."""
    logger.info(f"Начало обработки видео: {video_path}")
    started = time.time()

    def _p(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct}%] {msg}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    _p(5, "Загрузка модели...")
    pipe = _get_pipeline()

    _p(10, "Старт ML pipeline...")
    df = pipe.process_video(video_path, progress_callback=_p)

    _p(95, "Сохранение результатов и превью...")
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Сохранён CSV: {csv_path} ({len(df)} строк)")

    preview_path = output_dir / "preview.jpg"
    _generate_preview(video_path, df, preview_path)
    logger.info(f"Сохранён preview: {preview_path}")

    elapsed = time.time() - started
    _p(100, f"Готово за {elapsed:.1f} сек ({len(df)} ценников)")
    return {"dataframe": df, "csv_path": csv_path, "preview_path": preview_path}


def _generate_preview(video_path: Path, df: pd.DataFrame, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if len(df) > 0 and "frame_timestamp" in df.columns:
        ts = int(df.iloc[0]["frame_timestamp"])
        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH or h > PREVIEW_HEIGHT:
        scale = min(PREVIEW_WIDTH / w, PREVIEW_HEIGHT / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        scale = scale
    else:
        scale = 1.0

    if len(df) > 0:
        if "frame_timestamp" in df.columns:
            ts_show = int(df.iloc[0]["frame_timestamp"])
            rows = df[(df["frame_timestamp"] - ts_show).abs() < 500].head(6)
        else:
            rows = df.head(6)
        for _, r in rows.iterrows():
            try:
                x1 = int(int(r["x_min"]) * scale)
                y1 = int(int(r["y_min"]) * scale)
                x2 = int(int(r["x_max"]) * scale)
                y2 = int(int(r["y_max"]) * scale)
            except (ValueError, TypeError):
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
            label = str(r.get("price_card") or r.get("price_default") or "")
            if label and label != "nan":
                cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, TEXT_THICKNESS)

    cv2.putText(frame, "Lenta Tech — pricetag recognition",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
