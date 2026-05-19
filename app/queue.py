from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import sentry_sdk
import structlog
from celery import Celery, states
from celery.result import AsyncResult

from app.cleanup import cleanup_old_job_files
from app.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CLEANUP_INTERVAL_SECONDS,
    ENABLE_WORKER_METRICS,
    API_PUBLIC_URL,
    JOB_FILE_RETENTION_DAYS,
    PROMETHEUS_WORKER_PORT,
    WEBHOOK_TIMEOUT_SECONDS,
)
from app.db import get_job, init_db, update_job
from app.logging_config import configure_observability
from app.metrics import (
    JOBS_TOTAL,
    PROCESSING_TIME,
    RECOGNITION_RATE,
    calculate_recognition_rate,
    start_worker_metrics_server,
)
from app.pipeline import process_video
from app.progress import publish_job_progress

configure_observability()
logger = structlog.get_logger(__name__)
if ENABLE_WORKER_METRICS:
    start_worker_metrics_server(PROMETHEUS_WORKER_PORT)
init_db()

app = Celery(
    "pricetag",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=60 * 60 * 24,
    timezone="UTC",
    beat_schedule={
        "cleanup-old-job-files": {
            "task": "app.queue.cleanup_old_job_files_task",
            "schedule": CLEANUP_INTERVAL_SECONDS,
            "args": (JOB_FILE_RETENTION_DAYS,),
        },
    },
)


def _job_meta(
    status: str,
    progress: int,
    message: str,
    csv_path: str = "",
    preview_path: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "progress": progress,
        "message": message,
        "csv_path": csv_path,
        "preview_path": preview_path,
    }


def _db_job_meta(db_job: Dict[str, Any]) -> Dict[str, Any]:
    return _job_meta(
        db_job["status"],
        int(db_job["progress"]),
        db_job["message"],
        db_job["csv_path"],
        db_job["preview_path"],
    )


def _ready_results_meta(video: Path, job_dir: Path) -> Optional[Dict[str, Any]]:
    """Готовые артефакты на диске — чтобы не крутить пайплайн по дубликату из брокера."""
    results_csv = job_dir / "results.csv"
    preview_jpg = job_dir / "preview.jpg"
    if not (results_csv.is_file() and preview_jpg.is_file()):
        return None
    try:
        v_mtime = video.stat().st_mtime
        out_mtime = results_csv.stat().st_mtime
    except OSError:
        return None
    if out_mtime < v_mtime - 1e-3:
        return None
    return _job_meta(
        "done",
        100,
        "Обработка успешно завершена",
        csv_path=str(results_csv),
        preview_path=str(preview_jpg),
    )


@app.task(bind=True, name="app.queue.process_video_task")
def process_video_task(self, job_id: str, video_path: str, output_dir: str) -> Dict[str, Any]:
    video = Path(video_path)
    job_dir = Path(output_dir)
    started = time.monotonic()

    try:
        skip_meta = _ready_results_meta(video, job_dir)
        if skip_meta is not None:
            update_job(job_id, **skip_meta)
            publish_job_progress(job_id, skip_meta)
            self.update_state(state=states.SUCCESS, meta=skip_meta)
            logger.info(
                "job_skip_duplicate_broker_message",
                job_id=job_id,
                reason="results.csv уже есть (дубликат в очереди после успешного прогона)",
            )
            return skip_meta

        start_meta = _job_meta("processing", 0, "Начало обработки...")
        update_job(job_id, **start_meta)
        publish_job_progress(job_id, start_meta)
        self.update_state(
            state="PROGRESS",
            meta=start_meta,
        )

        def progress_callback(progress: int, message: str) -> None:
            progress_meta = _job_meta("processing", progress, message)
            update_job(job_id, **progress_meta)
            publish_job_progress(job_id, progress_meta)
            self.update_state(
                state="PROGRESS",
                meta=progress_meta,
            )
            logger.info("job_progress", job_id=job_id, progress=progress, message=message)

        result = process_video(video, job_dir, progress_callback)
        duration = time.monotonic() - started
        recognition_rate = calculate_recognition_rate(result.get("dataframe"))

        meta = _job_meta(
            "done",
            100,
            "Обработка успешно завершена",
            csv_path=str(result["csv_path"]),
            preview_path=str(result["preview_path"]),
        )
        update_job(job_id, **meta)
        JOBS_TOTAL.labels(status="done").inc()
        PROCESSING_TIME.observe(duration)
        RECOGNITION_RATE.labels(job_id=job_id).set(recognition_rate)
        publish_job_progress(job_id, meta)
        send_job_callback(job_id, meta)
        logger.info(
            "job_completed",
            job_id=job_id,
            duration_seconds=round(duration, 3),
            recognition_rate=round(recognition_rate, 4),
        )
        return meta
    except Exception as exc:
        duration = time.monotonic() - started
        logger.error("job_failed", job_id=job_id, error=str(exc), exc_info=True)
        sentry_sdk.capture_exception(exc)
        meta = _job_meta("error", 0, f"Ошибка обработки: {exc}")
        update_job(job_id, **meta)
        JOBS_TOTAL.labels(status="error").inc()
        PROCESSING_TIME.observe(duration)
        publish_job_progress(job_id, meta)
        send_job_callback(job_id, meta)
        self.update_state(
            state="ERROR",
            meta=meta,
        )
        return meta


@app.task(name="app.queue.cleanup_old_job_files_task")
def cleanup_old_job_files_task(retention_days: int = JOB_FILE_RETENTION_DAYS) -> Dict[str, int]:
    return cleanup_old_job_files(retention_days)


def revoke_job_task(job_id: str, terminate: bool = True) -> None:
    AsyncResult(job_id, app=app).revoke(terminate=terminate)


def send_job_callback(job_id: str, meta: Dict[str, Any]) -> None:
    job = get_job(job_id)
    callback_url = (job or {}).get("callback_url", "")

    if not callback_url:
        return

    result_url = ""
    if meta["status"] == "done":
        result_url = f"{API_PUBLIC_URL.rstrip('/')}/api/v2/jobs/{job_id}/result"

    payload = {
        "job_id": job_id,
        "status": meta["status"],
        "result_url": result_url,
        "message": meta.get("message", ""),
    }

    try:
        response = requests.post(callback_url, json=payload, timeout=WEBHOOK_TIMEOUT_SECONDS)
        response.raise_for_status()
        logger.info("webhook_callback_delivered", job_id=job_id, callback_url=callback_url)
    except requests.RequestException as exc:
        logger.warning(
            "webhook_callback_failed",
            job_id=job_id,
            callback_url=callback_url,
            error=str(exc),
        )


def get_task_status(job_id: str) -> Dict[str, Any]:
    result = AsyncResult(job_id, app=app)
    db_job = get_job(job_id)

    try:
        state = result.state
        result_info = result.info
        info = result_info if isinstance(result_info, dict) else {}
    except Exception as exc:
        if db_job:
            return _db_job_meta(db_job)
        return _job_meta("error", 0, f"Не удалось получить состояние задачи: {exc}")

    if state == states.PENDING:
        if db_job:
            return _db_job_meta(db_job)
        return _job_meta("queued", 0, "Видео загружено, ожидание обработки...")

    if state == states.STARTED:
        # Celery шлёт STARTED до первого update_state(PROGRESS); SQLite уже может
        # содержать реальный прогресс — иначе UI «обнуляется» на долгих шагах.
        if db_job and (
            db_job["status"] == "processing" or int(db_job["progress"] or 0) > 0
        ):
            return _db_job_meta(db_job)
        return _job_meta("processing", 0, "Начало обработки...")

    if state == "PROGRESS":
        prog = int(info.get("progress", 0))
        msg = info.get("message", "") or ""
        if db_job and prog == 0 and not msg and int(db_job.get("progress") or 0) > 0:
            return _db_job_meta(db_job)
        return _job_meta(
            info.get("status", "processing"),
            prog,
            msg,
            info.get("csv_path", ""),
            info.get("preview_path", ""),
        )

    if state == states.SUCCESS:
        return _job_meta(
            info.get("status", "done"),
            int(info.get("progress", 100)),
            info.get("message", "Обработка успешно завершена"),
            info.get("csv_path", ""),
            info.get("preview_path", ""),
        )

    if state in (states.FAILURE, "ERROR"):
        return _job_meta(
            info.get("status", "error"),
            int(info.get("progress", 0)),
            info.get("message", "Ошибка обработки"),
            info.get("csv_path", ""),
            info.get("preview_path", ""),
        )

    return _job_meta("processing", 0, f"Состояние задачи: {state}")
