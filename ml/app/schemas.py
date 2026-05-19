from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Уникальный идентификатор задачи")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Тип токена")
    role: Literal["viewer", "operator", "admin"] = Field(..., description="Роль пользователя")


class StatusResponse(BaseModel):
    status: Literal["queued", "processing", "done", "error"] = Field(
        ..., description="Текущий статус задачи"
    )
    progress: int = Field(..., ge=0, le=100, description="Процент прогресса обработки")
    message: str = Field(default="", description="Статусное сообщение или описание ошибки")


class JobMetadataResponse(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор задачи")
    status: Literal["queued", "processing", "done", "error"] = Field(
        ..., description="Текущий статус задачи"
    )
    progress: int = Field(..., ge=0, le=100, description="Процент прогресса обработки")
    message: str = Field(default="", description="Статусное сообщение или описание ошибки")
    csv_path: str = Field(default="", description="Путь к CSV с результатами")
    preview_path: str = Field(default="", description="Путь к preview-изображению")
    user_id: str = Field(default="anonymous", description="Идентификатор пользователя")
    callback_url: str = Field(default="", description="Webhook URL для callback после завершения")
    created_at: str = Field(..., description="Время создания задачи")
    updated_at: Optional[str] = Field(default=None, description="Время последнего обновления")


class JobsListResponse(BaseModel):
    jobs: List[JobMetadataResponse]


class DeleteJobResponse(BaseModel):
    id: str = Field(..., description="Идентификатор удаленной задачи")
    deleted: bool = Field(..., description="Флаг успешного удаления")
    message: str = Field(..., description="Результат операции")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Описание ошибки")
