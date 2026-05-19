from __future__ import annotations

import logging
import shutil
from typing import Dict

from app.config import JOBS_DIR, JOB_FILE_RETENTION_DAYS
from app.db import list_jobs_for_file_cleanup, mark_job_files_deleted

logger = logging.getLogger(__name__)


def cleanup_old_job_files(retention_days: int = JOB_FILE_RETENTION_DAYS) -> Dict[str, int]:
    jobs_root = JOBS_DIR.resolve()
    deleted_count = 0
    skipped_count = 0

    for job in list_jobs_for_file_cleanup(retention_days):
        job_id = job["id"]
        job_dir = (JOBS_DIR / job_id).resolve()

        if jobs_root not in job_dir.parents:
            logger.warning("Skip cleanup outside jobs dir: %s", job_dir)
            skipped_count += 1
            continue

        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
                deleted_count += 1

            mark_job_files_deleted(
                job_id,
                f"Файлы задачи удалены автоочисткой старше {retention_days} дней",
            )
        except Exception:
            logger.exception("Failed to cleanup job files: %s", job_id)
            skipped_count += 1

    logger.info(
        "Cleanup finished: deleted=%s skipped=%s retention_days=%s",
        deleted_count,
        skipped_count,
        retention_days,
    )
    return {"deleted": deleted_count, "skipped": skipped_count}
