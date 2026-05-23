FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Builder-only зависимости: компиляторы и cmake нужны для пакетов вроде
# llama-cpp-python на платформах без готовых wheels. В runtime они не попадут.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# Сначала torch CPU-only, чтобы pip не подтянул CUDA-зависимости.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1

COPY requirements.txt .
# pip увидит уже установленный torch и не будет тянуть CUDA-deps.
RUN pip install -r requirements.txt


FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    EASYOCR_MODEL_DIR=/opt/easyocr \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Runtime-only системные библиотеки:
#  - libgl1/libglib2.0-0/libsm6/libxext6/libxrender1 — для OpenCV
#  - libgomp1 — OpenMP для torch/EasyOCR CPU
#  - libzbar0 — pyzbar
#  - libmagic1 — MIME sniffing через python-magic
#  - curl — healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libzbar0 \
        libmagic1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# OCR-библиотеки подтягивают модели при первом создании объектов.
# Кэшируем их на этапе сборки, чтобы inference в контуре не зависел от сети.
RUN python - <<'PY'
import os
import easyocr
from paddleocr import PaddleOCR

try:
    easyocr.Reader(
        ["ru", "en"],
        gpu=False,
        verbose=False,
        model_storage_directory=os.environ["EASYOCR_MODEL_DIR"],
    )
except Exception as exc:
    print(f"WARNING: EasyOCR model prefetch failed: {exc}")
PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="ru",
    device="cpu",
)
PY

# приложение и ML код
COPY app/ ./app/
COPY ml/ ./ml/

# веса
# - ml/output/runs/pricetag_v2/weights/best.pt (~6 MB) — должны быть в COPY ml/
# - ml/weights/wechat_qr/ (~1 MB) — должны быть в COPY ml/
# - ml/weights/qwen2.5-3b-q4_k_m.gguf (~1.8 GB) — НЕ копируем в образ
#   (он монтируется как volume в docker-compose). Если volume не пробросить,
#   pipeline сам сделает graceful fallback: product_name остаётся пустым.

RUN mkdir -p /app/tmp/jobs

EXPOSE 8000 8501 8502 9101

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
