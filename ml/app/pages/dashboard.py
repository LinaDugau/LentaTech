import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "")

st.set_page_config(page_title="Pricetag Analytics Dashboard", layout="wide")


def main():
    st.title("Dashboard аналитики")
    st.caption("Операционная аналитика по обработанным видео и качеству распознавания")

    days = st.sidebar.slider("Период, дней", min_value=7, max_value=365, value=30, step=1)
    data = load_analytics(days)

    if not data:
        st.error("Не удалось загрузить аналитику")
        return

    render_metrics(data)
    render_daily_charts(data)
    render_distribution_charts(data)
    render_reporting_block(data)


@st.cache_data(ttl=60, show_spinner=False)
def load_analytics(days: int):
    try:
        response = requests.get(
            f"{API_URL}/api/v2/analytics/summary",
            params={"days": days},
            headers=auth_headers(),
            timeout=20,
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None


def auth_headers():
    if not API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {API_TOKEN}"}


def render_metrics(data: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Видео за неделю", data.get("processed_week", 0))
    col2.metric("Видео за месяц", data.get("processed_month", 0))
    col3.metric("Среднее время обработки, сек", data.get("avg_processing_time_seconds", 0))


def render_daily_charts(data: dict):
    st.subheader("Динамика по дням")
    jobs_by_day = pd.DataFrame(data.get("jobs_by_day", []))
    target_metric = pd.DataFrame(data.get("target_metric_by_day", []))

    left_col, right_col = st.columns(2)

    with left_col:
        st.caption("Количество обработанных видео")
        if jobs_by_day.empty:
            st.info("Нет данных по обработанным видео")
        else:
            st.line_chart(jobs_by_day, x="date", y="count")

    with right_col:
        st.caption("Средний TARGET_METRIC по дням")
        if target_metric.empty:
            st.info("Нет данных по TARGET_METRIC")
        else:
            st.line_chart(target_metric, x="date", y="target_metric")


def render_distribution_charts(data: dict):
    st.subheader("Распределения и топы")
    type_distribution = pd.DataFrame(data.get("price_tag_type_distribution", []))
    top_products = pd.DataFrame(data.get("top_products", []))

    left_col, right_col = st.columns(2)

    with left_col:
        st.caption("Распределение типов ценников")
        if type_distribution.empty:
            st.info("Нет данных по типам ценников")
        else:
            st.bar_chart(type_distribution, x="type", y="count")

    with right_col:
        st.caption("Топ-10 распознанных товаров")
        if top_products.empty:
            st.info("Нет распознанных товаров")
        else:
            st.dataframe(top_products, use_container_width=True, height=360)


def render_reporting_block(data: dict):
    st.subheader("Интеграция с Lenta-стеком")
    reporting = data.get("reporting", {})
    st.info(
        "Для ClickHouse / Greenplum подготовлен batch-friendly экспорт: "
        "`/result/{job_id}.parquet` и `/api/v2/jobs/{job_id}/detections`."
    )
    st.write(
        {
            "backend": reporting.get("backend", "parquet"),
            "recommended_format": reporting.get("recommended_format", "parquet"),
            "clickhouse_configured": reporting.get("clickhouse_configured", False),
            "greenplum_configured": reporting.get("greenplum_configured", False),
            "clickhouse": reporting.get("clickhouse", ""),
            "greenplum": reporting.get("greenplum", ""),
        }
    )


if __name__ == "__main__":
    main()
