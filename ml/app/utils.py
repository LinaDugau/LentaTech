from pathlib import Path
from typing import Optional
import uuid

import structlog

from app.config import ALLOWED_EXTENSIONS, JOBS_DIR
from app.logging_config import configure_observability

configure_observability()
logger = structlog.get_logger(__name__)


def generate_job_id() -> str:

    return str(uuid.uuid4())


def get_job_dir(job_id: str) -> Path:

    return JOBS_DIR / job_id


def create_job_dir(job_id: str) -> Path:

    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Создана директория задачи: {job_dir}")
    return job_dir


def validate_video_extension(filename: str) -> bool:

    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def get_file_size_mb(file_path: Path) -> float:

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def cleanup_job_files(job_id: str) -> None:

    job_dir = get_job_dir(job_id)
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
        logger.info(f"Очищена директория задачи: {job_dir}")
