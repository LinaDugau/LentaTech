# Запуск через Docker

Backend (FastAPI на :8000) + Streamlit UI (:8501) — два контейнера через
`docker-compose`.

## Что должно быть на диске перед запуском

```
lentatech/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── app/                                    ← backend код
├── ml/
│   ├── pipeline.py                         ← главный pipeline (используется backend'ом)
│   ├── scripts/                            ← все ML-скрипты
│   ├── output/runs/pricetag_v2/weights/best.pt   ← YOLO (~6 MB)  обязательно
│   └── weights/
│       ├── wechat_qr/                      ← WeChat QR (~1 MB)   опционально
│       └── qwen2.5-3b-q4_k_m.gguf          ← LLM (~1.8 GB)       опционально
```

Если каких-то опциональных весов нет — pipeline корректно работает без
них (соответствующие шаги пропускаются).

## Базовый запуск

```bash
# из корня проекта
docker compose up --build
```

После сборки (~5-10 мин первый раз):
- API:  http://localhost:8000  (swagger: http://localhost:8000/docs)
- Health: http://localhost:8000/health
- UI:   http://localhost:8501
- Dashboard: http://localhost:8502
- Prometheus API metrics: http://localhost:8000/metrics
- Prometheus worker metrics: http://localhost:9101/metrics

Загружаешь видео через UI → ждёшь обработку → скачиваешь `results.csv`.

## Что монтируется как volume

```yaml
volumes:
  - ./tmp:/app/tmp                  # video uploads + результаты заданий
  - ./ml/weights:/app/ml/weights    # LLM-веса (не копируются в образ)
  - ./ml/output:/app/ml/output      # intermediate ML файлы для persistence
```

То есть **LLM-веса** не пакуются в образ, а монтируются.
Это означает:
1. Dockerfile multi-stage: компиляторы, `cmake`, `ninja`, `build-essential`
   остаются только в builder stage, runtime-образ содержит готовый venv и
   минимальные системные библиотеки.
2. Образ остаётся заметно меньше, чем single-stage сборка с build toolchain и LLM.
   Целевой диапазон после исключения toolchain и больших весов: ~1.5-2 GB
   (зависит от torch/pyarrow/easyocr wheels и платформы).
3. Чтобы заменить модель — кладёшь новый `.pt`/`.gguf` в `ml/` и
   перезапускаешь контейнер (без пересборки).

## Полезные команды

```bash
# собрать + запустить в фоне
docker compose up --build -d

# логи API
docker compose logs -f api

# логи UI
docker compose logs -f ui

# остановить
docker compose down

# полный rebuild (если поменялся Dockerfile/requirements)
docker compose build --no-cache && docker compose up

# зайти внутрь api-контейнера
docker compose exec api bash

# проверить API живой
curl http://localhost:8000/

# загрузить видео через curl
curl -F "file=@Данные/43_15/43_15.mp4" http://localhost:8000/upload
# → {"job_id":"uuid..."}

# статус
curl http://localhost:8000/status/<job_id>

# результат
curl http://localhost:8000/result/<job_id>.csv -o result.csv
```

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `USE_LLM` | `1` | Подключать ли Qwen LLM для product_name. `0` = быстрее, но product_name пустой. |
| `API_HOST` | `0.0.0.0` | Адрес API |
| `API_PORT` | `8000` | Порт API |
| `OMP_NUM_THREADS` | `4` | Ограничение worker'ов PyTorch (CPU-only) |
| `API_URL` | `http://api:8000` | UI → API (внутри сети compose) |

Установить:
```bash
USE_LLM=0 docker compose up
# или в docker-compose.yml в секции environment
```

## Размеры и время

- **Сборка образа** первый раз: ~5-10 мин (зависит от сети, тянет torch ~2 GB).
- **Размер образа**: ~3-4 GB (без LLM).
- **Старт контейнера**: ~30-60 сек (модели грузятся в память при первом
  запросе, не на старте).
- **Обработка 1 видео (60 сек)**: ~30-50 мин на CPU. С GPU/MPS быстрее
  (но в Docker'е под Linux обычно только CPU).

## Health check

API имеет healthcheck (curl `/`). `start_period=120s` — даёт время
загрузить YOLO+EasyOCR+LLM перед началом проверок.

UI ждёт `api: condition: service_healthy` перед стартом — гарантия что
API готов принимать запросы.

## Если что-то не работает

### 1. Билд падает на `pyzbar` / нет libzbar
В Dockerfile уже стоит `libzbar0`. Если пересобирали `requirements.txt`
без Dockerfile — добавь `apt-get install libzbar0`.

### 2. EasyOCR долго грузится
Первый запуск — EasyOCR качает свои веса (~64 MB рус+англ модели) в
`~/.EasyOCR/`. В контейнере это `/root/.EasyOCR/` — можно смонтировать
volume чтобы не качал каждый раз:
```yaml
volumes:
  - easyocr_cache:/root/.EasyOCR
```

### 3. OOM (Out Of Memory)
Лимит в compose стоит 8 GB. На machine с малым RAM:
- уменьшить до 4 GB и установить `USE_LLM=0` (Qwen съедает 2-3 GB)
- или отдать pipeline'у только CPU без LLM:
  ```yaml
  environment:
    - USE_LLM=0
  ```

### 4. Очень медленно
- проверить что используется CPU (Docker под Linux обычно так)
- ускорить можно через `frame_step` в `ml/pipeline.py` (по умолчанию 3)
- альтернатива: `USE_LLM=0` — даст -10-15 минут на видео

### 5. Pipeline кидает «WeChat QR not loaded»
Не критично — это лог про опциональный шаг. Если хочется убрать —
скачай веса:
```bash
mkdir -p ml/weights/wechat_qr && cd ml/weights/wechat_qr
for f in detect.caffemodel detect.prototxt sr.caffemodel sr.prototxt; do
  curl -sL -o "$f" "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/refs/heads/wechat_qrcode/$f"
done
```

### 6. LLM не работает
Проверь что `ml/weights/qwen2.5-3b-q4_k_m.gguf` существует (~1.8 GB).
Если файла нет — pipeline просто пропустит этот шаг (логи: `LLM skipped: ...`).

## Тест на голом запуске

```bash
docker compose up --build -d
sleep 90  # дождаться загрузки моделей
curl -F "file=@Данные/43_15/43_15.mp4" \
  -F "user_id=anonymous" \
  -F "callback_url=https://example.com/lenta/webhook" \
  http://localhost:8000/api/v2/jobs
# → запоминаем job_id из ответа
watch -n 5 curl -s http://localhost:8000/api/v2/jobs/<job_id>
# real-time progress для внешнего UI через WebSocket
# websocat ws://localhost:8000/ws/jobs/<job_id>
# список прошлых задач хранится в SQLite
curl -s "http://localhost:8000/api/v2/jobs?user_id=anonymous&status=done&limit=10"
# когда status: done — забираем csv
curl http://localhost:8000/api/v2/jobs/<job_id>/result -o result.csv
curl http://localhost:8000/api/v2/jobs/<job_id>/detections -o detections.json
head result.csv
```

Swagger/OpenAPI доступен стандартно для FastAPI: `http://localhost:8000/docs`.
Dashboard аналитики доступен отдельно: `http://localhost:8502`.

Production JWT-auth включается через `AUTH_ENABLED=1`. Получить токен:

```bash
curl -X POST http://localhost:8000/api/v2/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=operator&password=operator"

curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v2/jobs
```

## Production-замечания (если хочешь деплоить)

1. Задачи выполняются через Celery + Redis: API кладёт job в очередь, отдельный
   `worker` запускает ML-пайплайн, прогресс и результат хранятся в Redis backend.
2. Real-time progress доступен через WebSocket `ws://<host>/ws/jobs/<job_id>`.
   Worker публикует обновления в Redis pub/sub (`REDIS_PROGRESS_URL`).
3. Метаданные задач дополнительно сохраняются в SQLite (`tmp/jobs.db`): история
   запусков не теряется при истечении Redis result backend.
4. Старые job-файлы очищаются через Celery Beat: `beat` раз в сутки ставит
   задачу очистки, а `worker` удаляет файлы завершённых/ошибочных задач старше
   `JOB_FILE_RETENTION_DAYS` дней (по умолчанию 7).
5. Если при upload указан `callback_url`, worker отправит POST после завершения:
   `{"job_id":"...","status":"done","result_url":"<API_PUBLIC_URL>/api/v2/jobs/.../result"}`.
   Для деплоя обязательно выставь внешний `API_PUBLIC_URL`.
6. Observability:
   - структурные JSON-логи через `structlog`;
   - Sentry включается переменной `SENTRY_DSN`;
   - Prometheus: `/metrics` на API и worker metrics server на `PROMETHEUS_WORKER_PORT`.
   Основные метрики: `pricetag_jobs_total`, `pricetag_processing_duration_seconds`,
   `pricetag_recognition_rate`.
7. Расширенный healthcheck: `/health` проверяет API, Redis, detector weights,
   наличие LLM-весов, loaded-state модели и свободный диск (`MIN_FREE_DISK_BYTES`,
   по умолчанию 2 GB). Docker healthcheck использует именно `/health`.
8. Production-auth: `AUTH_ENABLED=1` включает JWT для `/api/v2/*`.
   Роли: `viewer` читает статусы/результаты/аналитику, `operator` создаёт jobs,
   `admin` может удалять. Пользователи задаются JSON-переменной `JWT_USERS`,
   секрет обязательно заменить через `JWT_SECRET_KEY`.
9. Rate limiting включён через `slowapi`: upload/create-job ограничены
   `RATE_LIMIT_UPLOAD` (по умолчанию `5/minute`), login — `RATE_LIMIT_LOGIN`
   (по умолчанию `10/minute`). Отключить можно через `RATE_LIMIT_ENABLED=0`.
10. Upload validation:
   - расширение проверяется по allowlist `.mp4/.mov/.avi`;
   - MIME sniffing через `python-magic` и `libmagic1`;
   - длительность ограничена `MAX_VIDEO_DURATION_SECONDS` (по умолчанию 300 сек);
   - разрешение ограничено диапазоном `MIN_VIDEO_WIDTH/HEIGHT` и
     `MAX_VIDEO_WIDTH/HEIGHT` (по умолчанию 320x240..3840x2160).
11. Для масштабирования можно поднять несколько `worker`-контейнеров, но для
   CPU-only режима по умолчанию стоит `--concurrency=1`, чтобы не забить память.
12. Никакой авторизации нет только в демо-режиме `AUTH_ENABLED=0`, что соответствует
   требованию ТЗ «без дополнительной авторизации». Если будет публичный URL,
   включи JWT и добавь rate-limit.

## HuggingFace Spaces (для публичного демо)

`docker compose` напрямую туда не поднимается — Spaces ждёт один
Dockerfile. Самый простой путь:
1. Создать Space type=Docker.
2. Залить только Dockerfile + app/ + ml/ + requirements.txt.
3. В `Dockerfile` финальный `CMD streamlit run app/ui.py --server.port=7860`
   (Spaces ждёт 7860).
4. UI напрямую вызывает `process_video()` без FastAPI-прослойки — проще
   и быстрее в одном контейнере.
