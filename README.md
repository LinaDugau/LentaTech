# ShelfVision — распознавание ценников Lenta Tech

Команда **«Медвежата»**. Видео робота 4K → CSV с **29 полями** на каждый ценник (ТЗ Lenta Tech).

**Метрика:** `25/157` успешных ценников (≥80% из 23 содержательных полей).

| Видео | Успех | Примечание |
|---|---|---|
| 26_12-20 | 21/71 | barcode + QR + каталог |
| 43_15 | 4/29 | barcode-match |
| 25_12-20 | 0/57 | штрихкод не декодируется (blur/угол) |

Готовый submission: [`ml/submission/final_eval_*.csv`](ml/submission/).

---

## Быстрый старт (Docker)

Положите **`db_hack.csv`** от организаторов в корень репозитория (файл в `.gitignore`).

```bash
docker compose up --build -d
curl -fsS http://localhost:8000/health
```

| Сервис | URL |
|---|---|
| Upload UI | http://localhost:8501 |
| Dashboard | http://localhost:8502 |
| API / Swagger | http://localhost:8000/docs |

Загрузите `.mp4` в UI → дождитесь `status=done` (~30–50 мин на CPU) → скачайте CSV.

Подробности: [DOCKER_RUN.md](DOCKER_RUN.md).

---

## Pipeline

```
video → YOLO detect (v2+v4+v5b, TTA) → track → crops
     → OCR (EasyOCR + PaddleOCR) → QR/barcode decode
     → parse 29 полей → LLM product_name (Qwen2.5-3B)
     → match hybrid catalog (db_hack + scraped цены) → CSV
```

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

**Env (Docker / `app/pipeline.py`):** `USE_LLM`, `USE_PADDLE_OCR`, `USE_AGGRESSIVE_DECODE`, `ML_FRAME_STEP`, `ML_TOP_K`.

**29 колонок CSV:** `filename`, `product_name`, `price_default`, `price_card`, `price_discount`, `barcode`, `discount_amount`, `id_sku`, `print_datetime`, `code`, `additional_info`, `color`, `special_symbols`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max`, `qr_code_barcode`, `price1_qr`–`price4_qr`, `wholesale_level_1_count/price`, `wholesale_level_2_count/price`, `action_price_qr`, `action_code_qr`. Поле не предусмотрено типом ценника → `"нет"`; не удалось распознать → пусто.

**Веса:** YOLO `ml/output/runs/pricetag_v2/weights/best.pt` · Qwen `ml/weights/qwen2.5-3b-q4_k_m.gguf` · WeChat QR `ml/weights/wechat_qr/`. Без LLM — `use_llm=False`; без WeChat — соответствующий decode-проход пропускается.

**Системные deps:** `brew install zbar` (macOS) / `libzbar0` (Linux) для pyzbar; без zbar pipeline работает через WeChat + zxing + cv2.barcode.

---

## Offline-воспроизведение

Требуется: Python 3.11+, видео в `Данные/`, `db_hack.csv`, веса YOLO в `ml/output/runs/`,
опционально Qwen GGUF в `ml/weights/` (см. [DOCKER_RUN.md](DOCKER_RUN.md)).

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
python3 ml/scripts/evaluate_official.py
```

---

## Данные и обучение

| Источник | Назначение |
|---|---|
| `Данные/` — видео + GT от организаторов | eval, YOLO fine-tune |
| `Материалы/` — шаблоны ценников | synthetic dataset (5k кадров) |
| `db_hack.csv` — 604k SKU | product_name по barcode |
| lenta.com scraping (`scrape_lenta_undetected.py`) | цены для hybrid-каталога |
| Qwen2.5-3B-Q4 GGUF (Hugging Face) | product_name |

Ручной разметки нет. Инференс полностью автоматический.

Hybrid-каталог: `python3 ml/scripts/build_hybrid_catalog.py` → `ml/data/lenta_catalog_hybrid.parquet`.

### Скрипты в репозитории

**Production** (Docker / `ml/pipeline.py` / `app/pipeline.py`): `detect_and_track_tiled`, `extract_hires_crops`, `extract_qr_crops`, `run_ocr_qr`, `extra_bottom_ocr`, `paddle_full_ocr`, `decode_qr_crops`, `wechat_wholeframe`, `aggressive_decode`, `build_final_csv`, `llm_product_name`, `fill_product_name_fallback`, `match_to_catalog`, `final_postprocess`, `build_hybrid_catalog`.

**Offline-only:** `train_yolo*`, `build_gt_dataset*`, `evaluate_official`, `backup_weights`, `scrape_lenta_undetected`.

**Эксперименты** (не в prod-pipeline): `barcode_ml_infer`, `train_barcode_strip`, `sharpest_gated_decode`, `fuzzy_ocr_catalog`, `draw_bbox_overlay` и др.

---

## Ограничения

- **25_12-20:** barcode/QR не читаются декодерами → catalog-match невозможен.
- **Мелкий шрифт** (артикул, дата печати) теряется на blur-кадрах.
- **~50 мин/видео** на CPU; ускорение — GPU или RKNN int8 (RK3588).

---

## Стек

YOLOv8, EasyOCR, PaddleOCR, pyzbar, zxing-cpp, llama-cpp-python, FastAPI, Celery, Redis, Streamlit, Docker.

---

## Документы

- [DOCKER_RUN.md](DOCKER_RUN.md) — env, локальный запуск, Qwen weights
- [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md) — текст презентации

---

## Лицензия

Pre-trained веса (YOLOv8, Qwen, EasyOCR) — под лицензиями 
соответствующих
upstream-проектов (Apache-2.0 / MIT / Ultralytics AGPL-3.0 — 
см. их репозитории).
