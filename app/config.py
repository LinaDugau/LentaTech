import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TMP_DIR = BASE_DIR / "tmp"
JOBS_DIR = TMP_DIR / "jobs"

TMP_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

PROGRESS_UPDATE_INTERVAL = 0.5  
MOCK_PROCESSING_TIME = 10  

CSV_COLUMNS = ["timestamp", "product_name", "price", "barcode"]

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 3
TEXT_COLOR = (255, 255, 255) 
TEXT_THICKNESS = 2
