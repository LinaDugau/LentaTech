# ShelfVision — распознавание ценников Lenta Tech

Команда **«Медвежата»**. Видео робота 4K → CSV с **29 полями** на каждый ценник (ТЗ Lenta Tech).

**Метрика:** `26/157` успешных ценников (≥80% из 23 содержательных полей).

| Видео | Успех | Примечание |
|---|---|---|
| 26_12-20 | 22/71 | barcode + QR + каталог |
| 43_15 | 4/29 | barcode-match |
| 25_12-20 | 0/57 | штрихкод не декодируется (blur/угол) |

Готовый submission: [`ml/submission/final_eval_*.csv`](ml/submission/) — метрика **26/157** на этих файлах.

Docker и CLI используют **один** `ml/pipeline.py`: hybrid-каталог (`ml/data/lenta_catalog_hybrid.parquet`, db_hack + цены lenta.com) + `USE_SUBMISSION_FINALIZE=1`.

---

## Что положить локально

После `git clone` в репозитории уже есть YOLO (`ml/output/runs/pricetag_v2/weights/best.pt`) и WeChat QR (`ml/weights/wechat_qr/`). Остальное — с вашего диска:

| Файл / папка | Куда | Обязательно | Зачем |
|---|---|:---:|---|
| **`db_hack.csv`** | корень репо | ✅ | Каталог SKU от организаторов (~604k EAN). Без него нет match по barcode и имён из каталога |
| **`tmp/`** | корень репо | ✅ | Загруженные видео и CSV из UI/API. Создайте: `mkdir -p tmp` |
| **`ml/weights/qwen2.5-3b-q4_k_m.gguf`** | `ml/weights/` | ⚪ | LLM для `product_name` (~1.8 GB). Без файла pipeline работает, имена не заполняются |
| **`ml/data/lenta_catalog_hybrid.parquet`** | `ml/data/` | ⚪ | db_hack + цены lenta.com (~21 MB). Если нет — соберётся из `db_hack.csv` при первом прогоне |
| **`Данные/`** | корень репо | ⚪ | Видео и GT от организаторов — только для offline-eval и пошаговых скриптов (в git не кладём) |

**Минимум для Docker / UI:** `db_hack.csv` + `mkdir -p tmp`, затем `docker compose up --build -d`.

**Qwen (опционально):**
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-q4_k_m.gguf \
  --local-dir ml/weights
```

**Hybrid-каталог вручную** (если не хотите ждать auto-build при первом видео):
```bash
python3 ml/scripts/build_hybrid_catalog.py
# → ml/data/lenta_catalog_hybrid.parquet
```

**Системно (macOS):** `brew install zbar` — для pyzbar; без zbar остаются WeChat + zxing + cv2.barcode.

Подробнее про volumes, env и локальный Python без Docker: [DOCKER_RUN.md](DOCKER_RUN.md).

---

## Быстрый старт (Docker)

**Demo (VPS):** [UI — загрузка видео](http://158.160.229.185:8501) · [API / Swagger](http://158.160.229.185:8000/docs)

```bash
mkdir -p tmp
# положите db_hack.csv в корень репозитория
docker compose up --build -d
curl -fsS http://localhost:8000/health
```

| Сервис | Локально | Demo (VPS) |
|---|---|---|
| Upload UI | http://localhost:8501 | http://158.160.229.185:8501 |
| API / Swagger | http://localhost:8000/docs | http://158.160.229.185:8000/docs |

Загрузите `.mp4` в UI → дождитесь `status=done` (~30–50 мин на CPU) → скачайте CSV.

Подробности: [DOCKER_RUN.md](DOCKER_RUN.md).

---

## Pipeline

```
video → YOLO v2 (tiled 2×2) → track → crops
     → OCR (EasyOCR + PaddleOCR) → QR/barcode decode
     → parse 29 полей → LLM product_name (Qwen2.5-3B)
     → match по db_hack.csv → final_postprocess → CSV
```
*(Offline batch: ensemble v2+v4+v5b + TTA, hybrid parquet, apply_qr — для submission.)*

**Сервисы:** FastAPI + Celery + Redis + Streamlit. Всё локально, без облачных API.

### Python API

```python
from ml.pipeline import PriceTagPipeline

pipe = PriceTagPipeline()  # YOLO + LLM один раз (~5 сек)
df = pipe.process_video("/path/to/video.mp4")
df.to_csv("result.csv", index=False, encoding="utf-8")
```

CLI: `python3 ml/pipeline.py video.mp4 --out result.csv` · `--no_llm` · `--workdir /tmp/run`

Параметры конструктора: `weights`, `conf`, `imgsz`, `frame_step`, `top_k`, `use_llm`, `gguf_path`, `workdir`.

**Env (Docker / `ml/pipeline.py`):** `USE_LLM`, `USE_PADDLE_OCR`, `USE_AGGRESSIVE_DECODE`, `USE_SUBMISSION_FINALIZE`, `ML_FRAME_STEP`, `ML_TOP_K`.

**29 колонок CSV:** `filename`, `product_name`, `price_default`, `price_card`, `price_discount`, `barcode`, `discount_amount`, `id_sku`, `print_datetime`, `code`, `additional_info`, `color`, `special_symbols`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max`, `qr_code_barcode`, `price1_qr`–`price4_qr`, `wholesale_level_1_count/price`, `wholesale_level_2_count/price`, `action_price_qr`, `action_code_qr`. Поле не предусмотрено типом ценника → `"нет"`; не удалось распознать → пусто.

**Веса:** YOLO `ml/output/runs/pricetag_v2/weights/best.pt` · Qwen `ml/weights/qwen2.5-3b-q4_k_m.gguf` · WeChat QR `ml/weights/wechat_qr/`. Без LLM — `use_llm=False`; без WeChat — соответствующий decode-проход пропускается.

**Системные deps:** `brew install zbar` (macOS) / `libzbar0` (Linux) для pyzbar; без zbar pipeline работает через WeChat + zxing + cv2.barcode.

---

## Offline-воспроизведение

Нужны файлы из [«Что положить локально»](#что-положить-локально) + видео в `Данные/`. Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# detect → OCR → decode
python3 ml/scripts/detect_and_track_tiled.py --tta --conf 0.15
python3 ml/scripts/extract_hires_crops.py && python3 ml/scripts/extract_qr_crops.py
python3 ml/scripts/run_ocr_qr.py && python3 ml/scripts/paddle_full_ocr.py
python3 ml/scripts/decode_qr_crops.py && python3 ml/scripts/wechat_wholeframe.py
python3 ml/scripts/aggressive_decode.py

# catalog + final
python3 ml/scripts/build_hybrid_catalog.py
for v in 25_12-20 26_12-20 43_15; do
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv
  python3 ml/scripts/build_final_csv.py --ocr_csv ml/output/ocr_qr_$v.csv \
    --out ml/output/final_eval_$v.csv --keep_alts
  python3 ml/scripts/llm_product_name.py --final ml/output/final_eval_$v.csv \
    --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/fill_product_name_fallback.py --final ml/output/final_eval_$v.csv \
    --ocr ml/output/ocr_qr_$v.csv
  python3 ml/scripts/match_to_catalog.py --final ml/output/final_eval_$v.csv \
    --catalog ml/data/lenta_catalog_hybrid.parquet --threshold 0.65 --overwrite
done
python3 ml/scripts/final_postprocess.py --all
python3 ml/scripts/evaluate_official.py --pred-dir ml/submission
```

---

## Обучение и данные

Инференс полностью автоматический; ручная разметка использовалась **только для обучения**.

| Источник | Как использовали | Скрипты |
|---|---|---|
| `Данные/` — видео + GT от организаторов | eval, fine-tune YOLO, barcode strips | `analyze_videos.py`, `build_gt_dataset*.py`, `eval_against_gt.py` |
| `Материалы/` — шаблоны ценников | синтетический датасет (~5k кадров) | `synth_dataset.py`, `extract_backgrounds.py` |
| `db_hack.csv` — 604k SKU (организаторы) | match по EAN, product_name | `match_to_catalog.py`, `build_hybrid_catalog.py` |
| lenta.com scraping | цены для hybrid-каталога | `scrape_lenta_undetected.py` |
| YOLO-World / HSV pseudo-label | стартовая авторазметка | `pseudo_label.py`, `pseudo_label_color.py` |
| GT bbox + соседние кадры | датасет v2/v4/v5b | `build_gt_dataset.py`, `build_gt_dataset_extended.py` |
| Внешние barcode-датасеты (SBD и др.) | pretrain digit-strip CNN | `download_external_barcode_data.py`, `import_external_barcode_strips.py` |

**YOLO:** старт YOLOv8n → fine-tune **v2** (prod, tiled 2×2) → offline ensemble **v4/v5b** + TTA для submission-detect. Скрипты: `train_yolo.py`, `train_yolo_v4.py`, `train_yolo_v5.py`.

**Предобработка видео:** top-K кадров (sharpness × area), hires-crop pad 30%, QR-zone pad_top 120%, опционально undistort fisheye (`apply_undistort.py`, `example_undistort.py --preview`).

---

## Что пробовали

Prod-pipeline (`ml/pipeline.py`, Docker) и финальный submission **26/157** — это не лучший результат каждого эксперимента, а **стабильная** конфигурация без регрессий.

### Эволюция метрики

| Этап | TARGET | Комментарий |
|---|---|---|
| Baseline OCR + parse | ~4/157 | без каталога |
| + PaddleOCR, hybrid catalog | 25/157 | db_hack + lenta.com |
| + `apply_qr` при сборке submission | **26/157** | ✅ финальный submission |
| `run_name_boost` (fuzzy + LLM) | 25/157 | +45 имён, TARGET без прироста |
| dedup 2s | 24/157 | −30% строк, −1 успех |

### Эксперименты decode / barcode (не в prod)

| Скрипт / подход | TARGET | Verdict |
|---|---|---|
| `run_safe_boost` | 23/157 | ❌ heavy decode без catalog-gate |
| `run_heavy_boost` / `_sweep_safe_full` | 24/157 | ❌ регрессия на 26 |
| `run_aggressive_video_boost` (26 only) | 6/71 | ❌ rebuild final из OCR |
| `barcode_group_fusion` + Slot-CNN | 0 новых GT-barcode на 25 | blur/угол — физически не читается |
| `partial_ean_recovery` (10–12 цифр) | near-miss | без новых успехов на 25 |
| `barcode_zone_decode` + LANCZOS | offline | crop-only, без promote |
| fuzzy barcode без decode | — | ложные EAN → отклонено |

**Вывод:** узкое место — **чтение EAN на blur-кадрах (25_12-20)**. Из 26 успехов **24** — catalog-match по EAN. Heavy decode без catalog-gate ухудшает метрику.

### Catalog / product_name

| Подход | Скрипт | Эффект |
|---|---|---|
| barcode-match @0.65–0.70 | `match_to_catalog.py` | ключ №1 по ТЗ |
| fuzzy OCR → catalog | `fuzzy_ocr_catalog.py` | имена на near-threshold |
| LLM product_name | `llm_product_name.py` | Qwen2.5-3B локально |
| LLM rerank top-N SKU | `llm_catalog_rerank.py` | offline, без overwrite barcode |
| spatio-only (без EAN) | `spatio_catalog_resolve.py` | fallback, ниже порога 80% |

### Сводный sweep

```bash
python3 ml/scripts/run_metric_sweep.py   # только workdir, prod не трогает
python3 ml/scripts/run_top3_boost.py --videos 26_12-20  # zone + partial EAN
python3 ml/scripts/run_name_boost.py     # fuzzy name + LLM
```

---

## Скрипты в репозитории

**Production** (Docker / `ml/pipeline.py`):  
`detect_and_track_tiled`, `extract_hires_crops`, `extract_qr_crops`, `run_ocr_qr`, `extra_bottom_ocr`, `extra_top_ocr`, `paddle_full_ocr`, `decode_qr_crops`, `wechat_wholeframe`, `aggressive_decode`, `whole_frame_codes2`, `build_final_csv`, `llm_product_name`, `fill_product_name_fallback`, `match_to_catalog`, `final_postprocess`, `apply_qr_to_final`, `fuzzy_ocr_catalog`, `build_hybrid_catalog`.

**Offline batch** (сборка submission): шаги из [Offline-воспроизведение](#offline-воспроизведение) + `evaluate_official.py`.

**Обучение / датасеты:** `train_yolo*`, `build_gt_dataset*`, `pseudo_label*`, `synth_dataset.py`, `train_barcode_strip.py`, `build_barcode_strip_dataset.py`.

**Эксперименты** (не в prod, документируют итерации): `run_*boost`, `run_metric_sweep`, `barcode_*`, `partial_ean_recovery`, `llm_catalog_rerank`, `spatio_catalog_resolve`, `draw_bbox_overlay.py`, `filter_non_pricetag.py`, `near_miss_*`.

**Визуализация для презентации:**
```bash
python3 ml/scripts/draw_bbox_overlay.py --video 26_12-20
python3 example_undistort.py   # превью undistort на Данные/
python3 ml/scripts/analyze_videos.py
```

---

## Данные

См. таблицу [«Что положить локально»](#что-положить-локально). Hybrid-каталог: `python3 ml/scripts/build_hybrid_catalog.py` → `ml/data/lenta_catalog_hybrid.parquet`.

---

## Ограничения и масштабирование

**Текущие ограничения:**
- **25_12-20 (0/57):** barcode/QR не декодируются (blur) → catalog-match невозможен.
- **Мелкий шрифт** (id_sku, print_datetime) — почти не читается на motion blur.
- **Полный pipeline** (ensemble+TTA) — ~60–80 мин/видео на CPU, **24+ GB RAM** (OOM на слабом железе).
- **Детектор** — ~180 кадров одного типа; другие зоны требуют дообучения.
- **Каталог:** hybrid — цены только на части SKU; для прода нужен BIRD/API.

**Масштабирование (путь в магазин, не только демо):**

Под **масштабированием** понимаем: решение можно развернуть в контуре Ленты, адаптировать под другие стеллажи и форматы ценников, оптимизировать под ресурсы — без облачных API.

| Этап | Что делаем |
|------|------------|
| **Пилот (сейчас)** | Робот → сервер в контуре → CSV; Docker + Celery + UI |
| **Max quality** | Сервер 24 GB: v2+v4+v5b+TTA, hybrid, Qwen — как submission 26/157 |
| **Prod / edge** | RK3588 (RKNN int8): detect на роботе → crops на сервер → OCR + BIRD |
| **Качество** | BIRD вместо scrape; дообучение YOLO на 100+ видео разных зон; EAN-gate |
| **Скорость** | GPU на региональном сервере → 5–15 мин/видео |
| **Продукт** | CSV → задачи сотруднику; интеграция с контролем полки; ClickHouse/Greenplum |

Стационарная видеоаналитика полки в Ленте уже даёт **−40%** времени на выкладку в пилоте — робот с ценниками продолжает тот же сценарий по всему залу.

---

## Стек

YOLOv8, EasyOCR, PaddleOCR, pyzbar, zxing-cpp, llama-cpp-python, FastAPI, Celery, Redis, Streamlit, Docker.

---

## Документы

- [DOCKER_RUN.md](DOCKER_RUN.md) — env, локальный запуск, Qwen weights

---

## Лицензии

**Код этого репозитория** (ShelfVision) — без отдельного `LICENSE`-файла.

**Сторонние компоненты** — по лицензиям upstream-проектов:

| Компонент | Что используем | Лицензия | Ссылка |
|---|---|---|---|
| **Ultralytics YOLOv8** | `ultralytics`, fine-tuned `best.pt` | **AGPL-3.0** (коммерция — Enterprise License у Ultralytics) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **Qwen2.5-3B-Instruct** | GGUF для `product_name` | **Qwen Research License** (некоммерческое исследование / eval) | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **EasyOCR** | OCR ru+en | Apache-2.0 | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| **PaddleOCR / PaddlePaddle** | доп. OCR | Apache-2.0 | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| **OpenCV + WeChat QR** | decode, `ml/weights/wechat_qr/` | Apache-2.0 | [opencv_contrib/wechat_qrcode](https://github.com/opencv/opencv_contrib/tree/master/modules/wechat_qrcode) |
| **PyTorch** | inference | BSD-3-Clause | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **llama-cpp-python** | runtime Qwen GGUF | MIT | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **pyzbar / zxing-cpp** | barcode/QR decode | MIT / Apache-2.0 | [NaturalHistoryMuseum/pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar), [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) |
| **FastAPI, Celery, Streamlit** и др. из `requirements.txt` | backend | MIT / BSD / Apache-2.0 (см. PyPI) | — |

**Данные организаторов:** `db_hack.csv`, видео/GT в `Данные/` — материалы хакатона Lenta Tech, не распространяются в этом репозитории.

**Важно:**
- **Qwen** — не Apache-2.0; лицензия Alibaba Cloud разрешает **некоммерческое** использование (research/eval). Коммерческое — отдельное согласование с Alibaba Cloud.
- **YOLO (AGPL-3.0)** — при распространении производного ПО с `ultralytics`/обученными весами нужно открыть исходники на тех же условиях или купить Enterprise License у Ultralytics.
- Pre-trained веса EasyOCR/PaddleOCR скачиваются при первом запуске — на них действуют лицензии соответствующих репозиториев моделей.
