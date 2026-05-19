from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import sentry_sdk
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import uvicorn

from app.config import (
    MAX_FILE_SIZE_BYTES,
    API_HOST,
    API_PORT,
    RATE_LIMIT_LOGIN,
    RATE_LIMIT_UPLOAD,
)
from app.analytics import build_analytics_summary
from app.auth import (
    AuthUser,
    create_access_token,
    get_user_from_websocket_token,
    login_with_form,
    require_role,
)
from app.health import build_health_report
from app.limiter import limiter
from app.logging_config import configure_observability
from app.metrics import JOBS_TOTAL, render_prometheus_metrics
from app.db import create_job, delete_job, get_job, init_db, list_jobs, update_job
from app.schemas import (
    DeleteJobResponse,
    ErrorResponse,
    JobMetadataResponse,
    JobsListResponse,
    StatusResponse,
    TokenResponse,
    UploadResponse,
)
from app.utils import (
    generate_job_id,
    create_job_dir,
    validate_video_extension,
    get_job_dir,
    logger,
    cleanup_job_files,
)
from app.queue import get_task_status, process_video_task, revoke_job_task
from app.progress import get_progress_stream
from app.video_validation import validate_video_content, validate_video_file

configure_observability()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Запуск Video Processing API...")
    yield
    logger.info("Остановка Video Processing API...")


app = FastAPI(
    title="API Обработки Видео",
    description="API для обработки видео с ML pipeline",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)


@app.get("/")
async def root():
    return {
        "message": "API Обработки Видео",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "jobs": "GET /jobs",
            "api_v2_login": "POST /api/v2/auth/login",
            "api_v2_jobs": "POST /api/v2/jobs",
            "api_v2_job": "GET /api/v2/jobs/{id}",
            "api_v2_result": "GET /api/v2/jobs/{id}/result",
            "api_v2_preview": "GET /api/v2/jobs/{id}/preview",
            "api_v2_detections": "GET /api/v2/jobs/{id}/detections",
            "api_v2_analytics": "GET /api/v2/analytics/summary",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "websocket_progress": "WS /ws/jobs/{job_id}",
            "status": "GET /status/{job_id}",
            "result": "GET /result/{job_id}.csv",
            "result_xlsx": "GET /result/{job_id}.xlsx",
            "result_json": "GET /result/{job_id}.json",
            "result_parquet": "GET /result/{job_id}.parquet",
            "preview": "GET /preview/{job_id}.jpg",
            "video": "GET /video/{job_id}.mp4"
        }
    }


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    content, media_type = render_prometheus_metrics()
    return Response(content=content, media_type=media_type)


@app.get("/health", tags=["Health"], summary="Extended service healthcheck")
async def health() -> JSONResponse:
    report = build_health_report()
    status_code = 503 if report["status"] == "unhealthy" else 200
    return JSONResponse(content=report, status_code=status_code)


@app.post(
    "/api/v2/auth/login",
    response_model=TokenResponse,
    tags=["API v2 Auth"],
    summary="Login and receive JWT access token",
    responses={401: {"model": ErrorResponse}},
)
@limiter.limit(RATE_LIMIT_LOGIN)
async def api_v2_login(
    request: Request,
    user: AuthUser = Depends(login_with_form),
) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        token_type="bearer",
        role=user.role,
    )


async def create_video_job(
    file: UploadFile,
    user_id: str = "anonymous",
    callback_url: Optional[str] = None,
) -> UploadResponse:
    logger.info(f"Received upload request: {file.filename}")
    
    if not validate_video_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Неверный формат файла. Разрешены: .mp4, .mov, .avi"
        )

    job_id = generate_job_id()
    
    job_dir = create_job_dir(job_id)
    video_path = job_dir / "input.mp4"
    
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: 500 МБ"
            )

        mime_type = validate_video_content(content)
        
        with open(video_path, "wb") as f:
            f.write(content)

        video_meta = validate_video_file(video_path)
        
        logger.info(
            "video_upload_validated",
            path=str(video_path),
            size_mb=round(len(content) / 1024 / 1024, 2),
            mime_type=mime_type,
            duration_seconds=round(video_meta["duration_seconds"], 2),
            width=video_meta["width"],
            height=video_meta["height"],
        )
    except HTTPException:
        cleanup_job_files(job_id)
        raise
    except Exception as e:
        cleanup_job_files(job_id)
        logger.error(f"Не удалось сохранить видео: {str(e)}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить видео: {str(e)}")

    create_job(job_id, user_id=user_id, callback_url=callback_url)
    JOBS_TOTAL.labels(status="queued").inc()

    try:
        process_video_task.apply_async(
            args=[job_id, str(video_path), str(job_dir)],
            task_id=job_id,
        )
    except Exception as e:
        logger.error(f"Не удалось поставить задачу в очередь: {str(e)}")
        sentry_sdk.capture_exception(e)
        JOBS_TOTAL.labels(status="queue_error").inc()
        update_job(job_id, status="error", message="Очередь обработки временно недоступна")
        raise HTTPException(status_code=503, detail="Очередь обработки временно недоступна")

    logger.info(f"Создана задача {job_id}, отправлена в Celery")
    
    return UploadResponse(job_id=job_id)


@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    callback_url: Optional[str] = Form(default=None)
) -> UploadResponse:
    return await create_video_job(file, user_id, callback_url)


@app.get("/jobs", response_model=JobsListResponse)
async def get_jobs(
    user_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> JobsListResponse:
    return JobsListResponse(
        jobs=[
            JobMetadataResponse(**job)
            for job in list_jobs(user_id=user_id, status=status, limit=limit)
        ]
    )


@app.get("/status/{job_id}", response_model=StatusResponse, responses={404: {"model": ErrorResponse}})
async def get_status(job_id: str) -> StatusResponse:
    # Строка в SQLite необязательна: после очистки tmp/jobs.db прогресс живёт в Celery/Redis.
    job = get_task_status(job_id)
    return StatusResponse(
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    )


@app.websocket("/ws/jobs/{job_id}")
async def job_progress(
    websocket: WebSocket,
    job_id: str,
    token: Optional[str] = Query(default=None),
) -> None:
    await websocket.accept()

    try:
        get_user_from_websocket_token(token)
    except HTTPException as exc:
        await websocket.send_json(
            {"job_id": job_id, "status": "error", "progress": 0, "message": exc.detail}
        )
        await websocket.close(code=1008)
        return

    try:
        current_status = get_task_status(job_id)
        await websocket.send_json({"job_id": job_id, **current_status})

        if current_status["status"] in {"done", "error"}:
            await websocket.close()
            return

        async for update in get_progress_stream(job_id):
            await websocket.send_json({"job_id": job_id, **update})

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", job_id=job_id)


def get_completed_result_csv_path(job_id: str) -> Path:

    if not get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")
    
    job = get_task_status(job_id)
    
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {job['status']}"
        )
    
    csv_path = Path(job["csv_path"] or get_job_dir(job_id) / "results.csv")
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Файл результатов не найден")

    return csv_path


def ensure_export_file(job_id: str, export_format: str) -> Path:
    csv_path = get_completed_result_csv_path(job_id)
    export_path = get_job_dir(job_id) / f"results.{export_format}"

    if export_path.exists() and export_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return export_path

    df = pd.read_csv(csv_path)

    if export_format == "xlsx":
        df.to_excel(export_path, index=False, engine="openpyxl")
    elif export_format == "json":
        df.to_json(export_path, orient="records", force_ascii=False, indent=2)
    elif export_format == "parquet":
        df.to_parquet(export_path, index=False, engine="pyarrow")
    else:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: {export_format}")

    return export_path


def get_job_or_404(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")
    return job


def build_job_metadata_response(job_id: str) -> JobMetadataResponse:
    db_job = get_job_or_404(job_id)
    runtime = get_task_status(job_id)
    merged = {**db_job, **runtime, "id": job_id}
    return JobMetadataResponse(**merged)


@app.post(
    "/api/v2/jobs",
    response_model=UploadResponse,
    tags=["API v2 Jobs"],
    summary="Upload video and start recognition job",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def api_v2_create_job(
    request: Request,
    file: UploadFile = File(..., description="Видео с робота"),
    user_id: str = Form("anonymous", description="Идентификатор пользователя"),
    callback_url: Optional[str] = Form(
        default=None,
        description="Webhook URL, куда отправить POST после завершения задачи",
    ),
    current_user: AuthUser = Depends(require_role("operator")),
) -> UploadResponse:
    effective_user_id = user_id if user_id != "anonymous" else current_user.username
    return await create_video_job(file, effective_user_id, callback_url)


@app.get(
    "/api/v2/jobs",
    response_model=JobsListResponse,
    tags=["API v2 Jobs"],
    summary="List jobs with filters",
)
async def api_v2_list_jobs(
    user_id: Optional[str] = Query(default=None, description="Фильтр по пользователю"),
    status: Optional[str] = Query(default=None, description="Фильтр по статусу"),
    limit: int = Query(default=50, ge=1, le=200, description="Максимум задач в ответе"),
    current_user: AuthUser = Depends(require_role("viewer")),
) -> JobsListResponse:
    return JobsListResponse(
        jobs=[
            JobMetadataResponse(**job)
            for job in list_jobs(user_id=user_id, status=status, limit=limit)
        ]
    )


@app.get(
    "/api/v2/jobs/{job_id}",
    response_model=JobMetadataResponse,
    tags=["API v2 Jobs"],
    summary="Get job status and metadata",
    responses={404: {"model": ErrorResponse}},
)
async def api_v2_get_job(
    job_id: str,
    current_user: AuthUser = Depends(require_role("viewer")),
) -> JobMetadataResponse:
    return build_job_metadata_response(job_id)


@app.get(
    "/api/v2/jobs/{job_id}/result",
    tags=["API v2 Jobs"],
    summary="Download job result as CSV",
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def api_v2_get_result(
    job_id: str,
    current_user: AuthUser = Depends(require_role("viewer")),
) -> FileResponse:
    csv_path = get_completed_result_csv_path(job_id)

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"results_{job_id}.csv",
    )


@app.get(
    "/api/v2/jobs/{job_id}/preview",
    tags=["API v2 Jobs"],
    summary="Download preview image",
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def api_v2_get_preview(
    job_id: str,
    current_user: AuthUser = Depends(require_role("viewer")),
) -> FileResponse:
    get_job_or_404(job_id)
    job = get_task_status(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {job['status']}",
        )

    preview_path = Path(job["preview_path"] or get_job_dir(job_id) / "preview.jpg")

    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Изображение предпросмотра не найдено")

    return FileResponse(
        path=preview_path,
        media_type="image/jpeg",
        filename=f"preview_{job_id}.jpg",
    )


@app.get(
    "/api/v2/jobs/{job_id}/detections",
    tags=["API v2 Jobs"],
    summary="Get all detections as JSON records",
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def api_v2_get_detections(
    job_id: str,
    current_user: AuthUser = Depends(require_role("viewer")),
) -> dict:
    csv_path = get_completed_result_csv_path(job_id)
    df = pd.read_csv(csv_path)
    df = df.astype(object).where(pd.notna(df), None)
    detections = df.to_dict(orient="records")

    return {
        "job_id": job_id,
        "count": len(detections),
        "detections": detections,
    }


@app.get(
    "/api/v2/analytics/summary",
    tags=["API v2 Analytics"],
    summary="Get dashboard analytics summary",
)
async def api_v2_analytics_summary(
    days: int = Query(default=30, ge=1, le=365, description="Период агрегации в днях"),
    current_user: AuthUser = Depends(require_role("viewer")),
) -> dict:
    return build_analytics_summary(days)


@app.delete(
    "/api/v2/jobs/{job_id}",
    response_model=DeleteJobResponse,
    tags=["API v2 Jobs"],
    summary="Cancel and delete job",
    responses={404: {"model": ErrorResponse}},
)
async def api_v2_delete_job(
    job_id: str,
    current_user: AuthUser = Depends(require_role("admin")),
) -> DeleteJobResponse:
    get_job_or_404(job_id)
    revoke_job_task(job_id, terminate=True)
    cleanup_job_files(job_id)
    delete_job(job_id)
    JOBS_TOTAL.labels(status="deleted").inc()

    return DeleteJobResponse(
        id=job_id,
        deleted=True,
        message="Задача отменена/удалена, файлы очищены",
    )


@app.get("/result/{job_id}.csv", responses={404: {"model": ErrorResponse}})
async def get_result(job_id: str) -> FileResponse:

    csv_path = get_completed_result_csv_path(job_id)
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"results_{job_id}.csv"
    )


@app.get("/result/{job_id}.xlsx", responses={404: {"model": ErrorResponse}})
async def get_result_xlsx(job_id: str) -> FileResponse:
    xlsx_path = ensure_export_file(job_id, "xlsx")

    return FileResponse(
        path=xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"results_{job_id}.xlsx"
    )


@app.get("/result/{job_id}.json", responses={404: {"model": ErrorResponse}})
async def get_result_json(job_id: str) -> FileResponse:
    json_path = ensure_export_file(job_id, "json")

    return FileResponse(
        path=json_path,
        media_type="application/json",
        filename=f"results_{job_id}.json"
    )


@app.get("/result/{job_id}.parquet", responses={404: {"model": ErrorResponse}})
async def get_result_parquet(job_id: str) -> FileResponse:
    parquet_path = ensure_export_file(job_id, "parquet")

    return FileResponse(
        path=parquet_path,
        media_type="application/octet-stream",
        filename=f"results_{job_id}.parquet"
    )


@app.get("/preview/{job_id}.jpg", responses={404: {"model": ErrorResponse}})
async def get_preview(job_id: str) -> FileResponse:

    if not get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")
    
    job = get_task_status(job_id)
    
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {job['status']}"
        )
    
    preview_path = Path(job["preview_path"] or get_job_dir(job_id) / "preview.jpg")
    
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Изображение предпросмотра не найдено")
    
    return FileResponse(
        path=preview_path,
        media_type="image/jpeg",
        filename=f"preview_{job_id}.jpg"
    )


@app.get("/video/{job_id}.mp4", responses={404: {"model": ErrorResponse}})
async def get_video(job_id: str) -> FileResponse:

    if not get_job(job_id):
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")

    video_path = get_job_dir(job_id) / "input.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Исходное видео не найдено")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"input_{job_id}.mp4"
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
