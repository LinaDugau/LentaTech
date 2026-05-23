import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TMP_DIR = BASE_DIR / "tmp"
JOBS_DIR = TMP_DIR / "jobs"
JOBS_DB_PATH = TMP_DIR / "jobs.db"
DETECTOR_WEIGHTS_PATH = Path(
    os.getenv(
        "DETECTOR_WEIGHTS_PATH",
        str(BASE_DIR / "ml" / "output" / "runs" / "pricetag_v2" / "weights" / "best.pt"),
    )
)
LLM_WEIGHTS_PATH = Path(
    os.getenv(
        "LLM_WEIGHTS_PATH",
        str(BASE_DIR / "ml" / "weights" / "qwen2.5-3b-q4_k_m.gguf"),
    )
)
CATALOG_PATH = Path(
    os.getenv(
        "CATALOG_PATH",
        str(BASE_DIR / "db_hack.csv"),
    )
)
MIN_FREE_DISK_BYTES = int(os.getenv("MIN_FREE_DISK_BYTES", str(2 * 1024 * 1024 * 1024)))

TMP_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/avi",
}
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "300"))
MIN_VIDEO_WIDTH = int(os.getenv("MIN_VIDEO_WIDTH", "320"))
MIN_VIDEO_HEIGHT = int(os.getenv("MIN_VIDEO_HEIGHT", "240"))
MAX_VIDEO_WIDTH = int(os.getenv("MAX_VIDEO_WIDTH", "3840"))
MAX_VIDEO_HEIGHT = int(os.getenv("MAX_VIDEO_HEIGHT", "2160"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", f"http://localhost:{API_PORT}")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "local")
ENABLE_WORKER_METRICS = os.getenv("ENABLE_WORKER_METRICS", "0") in {"1", "true", "True"}
PROMETHEUS_WORKER_PORT = int(os.getenv("PROMETHEUS_WORKER_PORT", "9101"))

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
REDIS_PROGRESS_URL = os.getenv("REDIS_PROGRESS_URL", "redis://redis:6379/2")
WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "5"))

JOB_FILE_RETENTION_DAYS = int(os.getenv("JOB_FILE_RETENTION_DAYS", "7"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(24 * 60 * 60)))

REPORTING_BACKEND = os.getenv("REPORTING_BACKEND", "parquet")
CLICKHOUSE_DSN = os.getenv("CLICKHOUSE_DSN", "")
GREENPLUM_DSN = os.getenv("GREENPLUM_DSN", "")

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "0") in {"1", "true", "True"}
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "720"))
JWT_USERS = os.getenv(
    "JWT_USERS",
    '{"admin":{"password":"admin","role":"admin"}}',
)

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") in {"1", "true", "True"}
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")
RATE_LIMIT_LOGIN = os.getenv("RATE_LIMIT_LOGIN", "10/minute")

PROGRESS_UPDATE_INTERVAL = 0.5  
MOCK_PROCESSING_TIME = 10  

# 29 полей по ТЗ Lenta Tech (см. ml/pipeline.py FIELDS_PUBLIC)
CSV_COLUMNS = [
    "filename", "product_name", "price_default", "price_card",
    "price_discount", "barcode", "discount_amount", "id_sku",
    "print_datetime", "code", "additional_info", "color", "special_symbols",
    "frame_timestamp", "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255) 
TEXT_THICKNESS = 2
