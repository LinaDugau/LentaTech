from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.config import CLICKHOUSE_DSN, GREENPLUM_DSN, JOBS_DIR, REPORTING_BACKEND
from app.db import list_jobs_for_analytics

TARGET_METRIC_FIELDS = [
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


def build_analytics_summary(days: int = 30) -> Dict[str, Any]:
    jobs = list_jobs_for_analytics(days)
    jobs_df = pd.DataFrame(jobs)

    if jobs_df.empty:
        return _empty_summary(days)

    jobs_df["created_at_dt"] = pd.to_datetime(jobs_df["created_at"], errors="coerce")
    jobs_df["updated_at_dt"] = pd.to_datetime(jobs_df["updated_at"], errors="coerce")
    done_jobs = jobs_df[jobs_df["status"] == "done"].copy()

    detections_df = _load_detection_rows(done_jobs)

    return {
        "period_days": days,
        "processed_week": _count_since(done_jobs, 7),
        "processed_month": _count_since(done_jobs, 30),
        "avg_processing_time_seconds": _avg_processing_time(done_jobs),
        "target_metric_by_day": _target_metric_by_day(detections_df),
        "price_tag_type_distribution": _price_tag_type_distribution(detections_df),
        "top_products": _top_products(detections_df),
        "jobs_by_day": _jobs_by_day(done_jobs),
        "reporting": {
            "backend": REPORTING_BACKEND,
            "recommended_format": "parquet",
            "clickhouse_configured": bool(CLICKHOUSE_DSN),
            "greenplum_configured": bool(GREENPLUM_DSN),
            "clickhouse": "Use /result/{job_id}.parquet for batch ingestion into ClickHouse.",
            "greenplum": "Use /result/{job_id}.parquet or CSV COPY pipeline for Greenplum reporting.",
        },
    }


def _empty_summary(days: int) -> Dict[str, Any]:
    return {
        "period_days": days,
        "processed_week": 0,
        "processed_month": 0,
        "avg_processing_time_seconds": 0,
        "target_metric_by_day": [],
        "price_tag_type_distribution": [],
        "top_products": [],
        "jobs_by_day": [],
        "reporting": {
            "backend": REPORTING_BACKEND,
            "recommended_format": "parquet",
            "clickhouse_configured": bool(CLICKHOUSE_DSN),
            "greenplum_configured": bool(GREENPLUM_DSN),
            "clickhouse": "No completed jobs yet.",
            "greenplum": "No completed jobs yet.",
        },
    }


def _load_detection_rows(done_jobs: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for _, job in done_jobs.iterrows():
        csv_path = Path(str(job.get("csv_path") or ""))
        if not csv_path.exists():
            csv_path = JOBS_DIR / str(job["id"]) / "results.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df["job_id"] = job["id"]
        df["created_at"] = job["created_at"]
        if pd.isna(job["created_at_dt"]):
            continue
        df["created_day"] = job["created_at_dt"].date().isoformat()
        df["target_metric"] = df.apply(_target_metric_proxy, axis=1)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _target_metric_proxy(row: pd.Series) -> float:
    available = 0
    total = 0

    for field in TARGET_METRIC_FIELDS:
        if field not in row:
            continue
        total += 1
        if not _is_empty(row[field]):
            available += 1

    return round(available / total, 4) if total else 0.0


def _is_empty(value: Any) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _count_since(done_jobs: pd.DataFrame, days: int) -> int:
    if done_jobs.empty:
        return 0
    since = pd.Timestamp(datetime.utcnow() - timedelta(days=days))
    return int((done_jobs["created_at_dt"] >= since).sum())


def _avg_processing_time(done_jobs: pd.DataFrame) -> float:
    if done_jobs.empty:
        return 0
    durations = (done_jobs["updated_at_dt"] - done_jobs["created_at_dt"]).dt.total_seconds()
    durations = durations.dropna()
    return round(float(durations.mean()), 2) if not durations.empty else 0


def _target_metric_by_day(detections_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if detections_df.empty:
        return []
    grouped = detections_df.groupby("created_day")["target_metric"].mean().reset_index()
    grouped["target_metric"] = grouped["target_metric"].round(4)
    return grouped.rename(columns={"created_day": "date"}).to_dict(orient="records")


def _price_tag_type_distribution(detections_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if detections_df.empty:
        return []

    type_series = pd.Series(["unknown"] * len(detections_df), index=detections_df.index)
    if "color" in detections_df.columns:
        type_series = detections_df["color"].where(~detections_df["color"].apply(_is_empty), type_series)
    if "special_symbols" in detections_df.columns:
        type_series = type_series.where(
            type_series.astype(str) != "unknown",
            detections_df["special_symbols"].where(
                ~detections_df["special_symbols"].apply(_is_empty),
                "unknown",
            ),
        )

    counts = type_series.astype(str).value_counts().head(20).reset_index()
    counts.columns = ["type", "count"]
    return counts.to_dict(orient="records")


def _top_products(detections_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if detections_df.empty or "product_name" not in detections_df.columns:
        return []

    products = detections_df["product_name"].dropna().astype(str).str.strip()
    products = products[(products != "") & (products.str.lower() != "nan")]
    counts = products.value_counts().head(10).reset_index()
    counts.columns = ["product_name", "count"]
    return counts.to_dict(orient="records")


def _jobs_by_day(done_jobs: pd.DataFrame) -> List[Dict[str, Any]]:
    if done_jobs.empty:
        return []

    grouped = done_jobs.groupby(done_jobs["created_at_dt"].dt.date).size().reset_index(name="count")
    grouped.columns = ["date", "count"]
    grouped["date"] = grouped["date"].astype(str)
    return grouped.to_dict(orient="records")
