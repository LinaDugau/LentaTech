from __future__ import annotations

from typing import Optional

import pandas as pd
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

JOBS_TOTAL = Counter("pricetag_jobs_total", "Total jobs", ["status"])
PROCESSING_TIME = Histogram(
    "pricetag_processing_duration_seconds",
    "Video processing duration in seconds",
)
RECOGNITION_RATE = Gauge(
    "pricetag_recognition_rate",
    "Proxy recognition rate for completed jobs",
    ["job_id"],
)

_METRICS_SERVER_STARTED = False


def start_worker_metrics_server(port: int) -> None:
    global _METRICS_SERVER_STARTED
    if _METRICS_SERVER_STARTED:
        return
    start_http_server(port)
    _METRICS_SERVER_STARTED = True


def render_prometheus_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def calculate_recognition_rate(df: Optional[pd.DataFrame]) -> float:
    if df is None or df.empty:
        return 0.0

    important_columns = [
        "product_name",
        "price_default",
        "price_card",
        "barcode",
        "id_sku",
        "frame_timestamp",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    existing_columns = [col for col in important_columns if col in df.columns]
    if not existing_columns:
        return 0.0

    filled = df[existing_columns].notna() & (df[existing_columns].astype(str).apply(lambda s: s.str.strip()) != "")
    return float(filled.sum().sum() / (len(df) * len(existing_columns)))
