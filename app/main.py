import logging
from pathlib import Path
from typing import Dict, Any
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import (
    MAX_FILE_SIZE_BYTES,
    API_HOST,
    API_PORT
)
from app.schemas import UploadResponse, StatusResponse, ErrorResponse
from app.utils import (
    generate_job_id,
    create_job_dir,
    validate_video_extension,
    get_job_dir,
    logger
)
from app.pipeline import process_video

jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск Video Processing API...")
    yield
    logger.info("Остановка Video Processing API...")


app = FastAPI(
    title="API Обработки Видео",
    description="API для обработки видео с ML pipeline",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def update_job_progress(job_id: str, progress: int, message: str) -> None:
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"] = message
        logger.info(f"Job {job_id}: {progress}% - {message}")


async def process_video_task(job_id: str, video_path: Path, output_dir: Path) -> None:

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = "Начало обработки..."

        def progress_callback(progress: int, message: str):
            update_job_progress(job_id, progress, message)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_video,
            video_path,
            output_dir,
            progress_callback
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Обработка успешно завершена"
        jobs[job_id]["csv_path"] = str(result["csv_path"])
        jobs[job_id]["preview_path"] = str(result["preview_path"])
        
        logger.info(f"Задача {job_id} успешно завершена")
        
    except Exception as e:
        logger.error(f"Задача {job_id} не удалась: {str(e)}", exc_info=True)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"Ошибка обработки: {str(e)}"


@app.get("/")
async def root():
    return {
        "message": "API Обработки Видео",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "status": "GET /status/{job_id}",
            "result": "GET /result/{job_id}.csv",
            "preview": "GET /preview/{job_id}.jpg"
        }
    }


@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
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
        
        with open(video_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved video file: {video_path} ({len(content) / 1024 / 1024:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Не удалось сохранить видео: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить видео: {str(e)}")
    
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Видео загружено, ожидание обработки...",
        "csv_path": "",
        "preview_path": ""
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, job_id, video_path, job_dir)
    
    logger.info(f"Создана задача {job_id}, запущена фоновая обработка")
    
    return UploadResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=StatusResponse, responses={404: {"model": ErrorResponse}})
async def get_status(job_id: str) -> StatusResponse:

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = jobs[job_id]
    
    return StatusResponse(
        status=job["status"],
        progress=job["progress"],
        message=job["message"]
    )

@app.get("/result/{job_id}.csv", responses={404: {"model": ErrorResponse}})
async def get_result(job_id: str) -> FileResponse:

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")
    
    job = jobs[job_id]
    
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {job['status']}"
        )
    
    csv_path = Path(job["csv_path"])
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Файл результатов не найден")
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"results_{job_id}.csv"
    )


@app.get("/preview/{job_id}.jpg", responses={404: {"model": ErrorResponse}})
async def get_preview(job_id: str) -> FileResponse:

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Задача не найдена: {job_id}")
    
    job = jobs[job_id]
    
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {job['status']}"
        )
    
    preview_path = Path(job["preview_path"])
    
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Изображение предпросмотра не найдено")
    
    return FileResponse(
        path=preview_path,
        media_type="image/jpeg",
        filename=f"preview_{job_id}.jpg"
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
