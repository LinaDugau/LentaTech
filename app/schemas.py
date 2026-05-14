from typing import Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Уникальный идентификатор задачи")


class StatusResponse(BaseModel):
    status: Literal["queued", "processing", "done", "error"] = Field(
        ..., description="Текущий статус задачи"
    )
    progress: int = Field(..., ge=0, le=100, description="Процент прогресса обработки")
    message: str = Field(default="", description="Статусное сообщение или описание ошибки")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Описание ошибки")
