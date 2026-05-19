from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import HTTPException

try:
    import magic
except ImportError:
    magic = None

from app.config import (
    ALLOWED_VIDEO_MIME_TYPES,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_HEIGHT,
    MAX_VIDEO_WIDTH,
    MIN_VIDEO_HEIGHT,
    MIN_VIDEO_WIDTH,
)


def validate_video_content(content: bytes) -> str:
    if magic is None:
        raise HTTPException(
            status_code=500,
            detail="MIME validation is unavailable: python-magic/libmagic is not installed",
        )

    try:
        mime_type = magic.from_buffer(content[:4096], mime=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось определить MIME-type: {exc}") from exc

    if mime_type not in ALLOWED_VIDEO_MIME_TYPES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_MIME_TYPES))
        raise HTTPException(
            status_code=400,
            detail=f"Неверный MIME-type видео: {mime_type}. Разрешены: {allowed}",
        )

    return mime_type


def validate_video_file(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Видео не открывается или повреждено")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if fps <= 0 or frame_count <= 0:
            raise HTTPException(status_code=400, detail="Не удалось определить длительность видео")

        duration_seconds = frame_count / fps
        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Видео слишком длинное: {duration_seconds:.1f} сек. "
                    f"Максимум: {MAX_VIDEO_DURATION_SECONDS} сек"
                ),
            )

        if width < MIN_VIDEO_WIDTH or height < MIN_VIDEO_HEIGHT:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Разрешение видео слишком маленькое: {width}x{height}. "
                    f"Минимум: {MIN_VIDEO_WIDTH}x{MIN_VIDEO_HEIGHT}"
                ),
            )

        if width > MAX_VIDEO_WIDTH or height > MAX_VIDEO_HEIGHT:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Разрешение видео слишком большое: {width}x{height}. "
                    f"Максимум: {MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT}"
                ),
            )

        return {
            "duration_seconds": duration_seconds,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()
