"""Backend pipeline — обёртка над `ml.pipeline.PriceTagPipeline`.

Раньше здесь был mock, генерирующий случайные товары. Теперь это реальная
интеграция с ML-pipeline (v15 stack).

API остался прежним:
    process_video(video_path, output_dir, progress_callback) -> dict

Возвращает:
    {"dataframe": pd.DataFrame, "csv_path": Path, "preview_path": Path}
"""
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

# pyzbar требует zbar — путь на macOS через brew, на Linux libzbar0
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

# ленивый синглтон — модель грузится один раз на процесс
_PIPELINE: Any = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        # импорт здесь, чтобы не тянуть ultralytics при import-only сценариях
        from ml.pipeline import PriceTagPipeline

        # use_llm=True по умолчанию. Если веса LLM не лежат, PriceTagPipeline
        # внутри сделает graceful fallback.
        use_llm = os.getenv("USE_LLM", "1") not in ("0", "false", "False")
        use_paddle = os.getenv("USE_PADDLE_OCR", "1") not in ("0", "false", "False")
        use_aggressive = os.getenv("USE_AGGRESSIVE_DECODE", "1") not in ("0", "false", "False")
        frame_step = int(os.getenv("ML_FRAME_STEP", "3"))
        top_k = int(os.getenv("ML_TOP_K", "3"))
        logger.info(
            "Loading PriceTagPipeline "
            f"(use_llm={use_llm}, weights={DETECTOR_WEIGHTS_PATH}, "
            f"frame_step={frame_step}, top_k={top_k}, "
            f"use_paddle={use_paddle}, use_aggressive={use_aggressive}, "
            f"catalog={CATALOG_PATH})..."
        )
        _PIPELINE = PriceTagPipeline(
            weights=DETECTOR_WEIGHTS_PATH,
            frame_step=frame_step,
            top_k=top_k,
            use_llm=use_llm,
            gguf_path=LLM_WEIGHTS_PATH,
            use_paddle=use_paddle,
            use_aggressive_decode=use_aggressive,
            catalog_path=CATALOG_PATH,
        )
        logger.info("PriceTagPipeline loaded")
    return _PIPELINE


def process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Запускает ML-pipeline по шагам с обновлением прогресса между ними.

    Шаги (приблизительная разбивка времени на 60-сек видео):
      5%   load model
      15%  tiled detect + track (~5 мин)
      25%  hi-res crops (~30 сек)
      30%  qr crops (~30 сек)
      60%  OCR (~30 мин)   ← bottleneck
      70%  QR decode (~3 прохода, ~10 мин)
      80%  build CSV (~10 сек)
      95%  LLM enrichment (~3 мин)
      100% preview + save
    """
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

    # импорты внутренних шагов из ml.pipeline
    from ml.pipeline import _move_video_to_data_dir
    from ml.scripts.extract_hires_crops import process as run_hires_extract
    from ml.scripts.extract_qr_crops import process as run_qr_extract
    from ml.scripts.run_ocr_qr import process_summary as run_ocr_summary
    from ml.scripts.extra_bottom_ocr import process as run_extra_bottom_ocr
    from ml.scripts.decode_qr_crops import process as run_qr_decode
    from ml.scripts.wechat_wholeframe import process as run_wholeframe
    from ml.scripts.aggressive_decode import process as run_aggressive_decode
    import tempfile
    import shutil

    # подготавливаем рабочее место. Все скрипты ml/ пишут в фиксированный
    # ROOT/ml/output, используем именно его, чтобы пути совпадали.
    from ml.pipeline import ROOT as ML_ROOT
    local_video, tmp_data_dir = _move_video_to_data_dir(video_path)
    workdir = ML_ROOT / "ml" / "output"
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) tiled detect + track
        _p(15, "Детектор + трекинг (~5 мин)...")
        pipe._step_detect(local_video, out_dir)
        stem = local_video.stem
        summary_csv = out_dir / f"track_summary_{stem}.csv"

        # 2) hi-res crops
        _p(25, "Вырезка кадров ценников...")
        run_hires_extract(summary_csv, pad_pct=0.30)
        hires_csv = out_dir / f"track_summary_hires_{stem}.csv"

        # 3) QR-crops (расширенные, QR обычно над ценником)
        _p(30, "Вырезка QR-зон...")
        run_qr_extract(hires_csv, pad_x=0.40,
                       pad_y_top=1.20, pad_y_bot=0.40)
        qr_summary_csv = out_dir / f"track_summary_qr_{stem}.csv"

        # 4) OCR (самый долгий шаг)
        _p(40, "OCR текста на каждом кадре (~30 мин)...")
        ocr_csv = run_ocr_summary(hires_csv)

        _p(65, "OCR нижней зоны ценника...")
        try:
            run_extra_bottom_ocr(ocr_csv)
        except Exception as e:
            logger.warning(f"extra_bottom_ocr skipped: {e}")
        if pipe.use_paddle:
            _p(68, "PaddleOCR: дополнительное чтение ценника (отдельный процесс)...")
            pipe._paddle_enrich_ocr(ocr_csv)

        # 5) QR/штрихкод декодинг — 3 прохода
        _p(70, "Декодинг штрихкодов: WeChat на crop'ах...")
        try:
            run_qr_decode(qr_summary_csv, ocr_csv)
        except Exception as e:
            logger.warning(f"decode_qr_crops skipped: {e}")
        _p(75, "Декодинг: WeChat на полном кадре...")
        try:
            run_wholeframe(ocr_csv)
        except Exception as e:
            logger.warning(f"wholeframe skipped: {e}")
        _p(80, "Декодинг: aggressive (preprocessing + 4 декодера)...")
        if pipe.use_aggressive_decode:
            try:
                run_aggressive_decode(ocr_csv)
            except Exception as e:
                logger.warning(f"aggressive_decode skipped: {e}")
        else:
            logger.info("aggressive_decode disabled by USE_AGGRESSIVE_DECODE=0")

        # 6) сборка финального DataFrame
        _p(85, "Сборка CSV (парсер полей)...")
        df = pipe._build_final(ocr_csv, keep_alts=False)

        # 7) LLM enrichment
        if pipe.use_llm:
            _p(90, "LLM: извлечение названий товаров...")
            pipe._llm_enrich(df, ocr_csv)
        _p(92, "Каталог: SKU, QR-поля и fallback названий...")
        pipe._post_build_enrich(df, ocr_csv)

        # 8) save + preview
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
    finally:
        # workdir = ROOT/ml/output (shared), не удаляем
        try:
            shutil.rmtree(tmp_data_dir, ignore_errors=True)
        except Exception:
            pass


def _generate_preview(video_path: Path, df: pd.DataFrame, output_path: Path) -> None:
    """Рисует bbox первых 5 ценников на одном из ранних кадров."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    # берём момент в районе первой pred-строки если есть
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
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        # пересчитаем bbox после ресайза
    else:
        scale = 1.0

    # рисуем bbox первых ценников на этом кадре
    if len(df) > 0:
        # выбираем строки с frame_timestamp близким к показанному кадру
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
