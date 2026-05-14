import time
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any
import random

import cv2
import numpy as np
import pandas as pd

from app.config import (
    CSV_COLUMNS,
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT,
    BBOX_COLOR,
    BBOX_THICKNESS,
    TEXT_COLOR,
    TEXT_THICKNESS,
    MOCK_PROCESSING_TIME,
    PROGRESS_UPDATE_INTERVAL
)

logger = logging.getLogger(__name__)


def process_video(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    logger.info(f"Начало обработки видео: {video_path}")
    
    try:
        if progress_callback:
            progress_callback(0, "Инициализация pipeline...")

        time.sleep(PROGRESS_UPDATE_INTERVAL)
        if progress_callback:
            progress_callback(10, "Загрузка видео файла...")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Свойства видео: {frame_count} кадров, {fps} fps, {duration:.2f}s длительность")
        
        if progress_callback:
            progress_callback(20, "Анализ кадров видео...")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.warning("Не удалось прочитать кадр, генерация пустого предпросмотра")
            frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        
        stages = [
            (30, "Обнаружение объектов..."),
            (50, "Распознавание товаров..."),
            (70, "Извлечение штрих-кодов..."),
            (85, "Генерация результатов..."),
        ]
        
        for progress, message in stages:
            time.sleep(MOCK_PROCESSING_TIME / len(stages))
            if progress_callback:
                progress_callback(progress, message)
        
        results_data = generate_mock_results(duration)
        df = pd.DataFrame(results_data, columns=CSV_COLUMNS)
        
        if progress_callback:
            progress_callback(90, "Сохранение результатов...")
        
        csv_path = output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Сохранены CSV результаты: {csv_path}")
        
        preview_path = output_dir / "preview.jpg"
        generate_preview_image(frame, results_data[:3], preview_path)
        logger.info(f"Сохранено изображение предпросмотра: {preview_path}")
        
        if progress_callback:
            progress_callback(100, "Обработка завершена!")
        
        logger.info("Обработка видео успешно завершена")
        
        return {
            "dataframe": df,
            "csv_path": csv_path,
            "preview_path": preview_path
        }
        
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {str(e)}", exc_info=True)
        raise


def generate_mock_results(duration: float, num_detections: int = 15) -> list:
    products = [
        ("Coca-Cola 0.5L", "1.99", "5449000000996"),
        ("Snickers Bar", "0.89", "5000159461122"),
        ("Red Bull 250ml", "2.49", "9002490100070"),
        ("Lay's Chips Classic", "2.29", "8710398510006"),
        ("Milka Chocolate", "1.79", "7622210449283"),
        ("Haribo Goldbears", "1.49", "4001686301012"),
        ("Pringles Original", "2.99", "5053990101566"),
        ("Kit Kat 4 Finger", "0.99", "5000159459228"),
        ("Sprite 0.5L", "1.89", "5449000000439"),
        ("Mars Bar", "0.85", "5000159407236"),
    ]
    
    results = []
    for i in range(num_detections):
        timestamp = round(random.uniform(0, max(duration, 10)), 2)
        product = random.choice(products)
        
        results.append([
            f"{timestamp:.2f}s",
            product[0],
            f"${product[1]}",
            product[2]
        ])
    
    results.sort(key=lambda x: float(x[0].replace('s', '')))
    
    return results


def generate_preview_image(
    frame: np.ndarray,
    detections: list,
    output_path: Path
) -> None:
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH or h > PREVIEW_HEIGHT:
        scale = min(PREVIEW_WIDTH / w, PREVIEW_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w
    
    preview = frame.copy()
    
    if preview.mean() < 10:
        preview = np.ones((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8) * 50
        h, w = PREVIEW_HEIGHT, PREVIEW_WIDTH
    
    num_boxes = min(len(detections), 3)
    for i in range(num_boxes):
        box_w, box_h = random.randint(100, 200), random.randint(80, 150)
        x1 = random.randint(50, max(51, w - box_w - 50))
        y1 = random.randint(50, max(51, h - box_h - 50))
        x2, y2 = x1 + box_w, y1 + box_h
        
        cv2.rectangle(preview, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
        
        product_name = detections[i][1] if i < len(detections) else "Product"
        label = f"{product_name}"
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_THICKNESS
        )
        
        cv2.rectangle(
            preview,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            BBOX_COLOR,
            -1
        )
        
        cv2.putText(
            preview,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            TEXT_THICKNESS
        )
    
    title = "Предпросмотр обнаружения товаров"
    cv2.putText(
        preview,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 0),
        3
    )
    
    cv2.imwrite(str(output_path), preview)
    logger.info(f"Сгенерирован предпросмотр с {num_boxes} рамками обнаружения")
