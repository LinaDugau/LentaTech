import time
import os
import io
from pathlib import Path
import tempfile

import cv2
import requests
import streamlit as st
import pandas as pd
from PIL import Image

from app.config import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB

API_URL = os.getenv("API_URL", "http://localhost:8000")
PRICE_COLUMNS = ["price_default", "price_card", "price_discount"]
COMPARE_KEY_COLUMNS = ["id_sku", "barcode"]
MANDATORY_COLUMNS = [
    "product_name",
    "barcode",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]

st.set_page_config(
    page_title="ML Обработка Видео",
    layout="wide"
)


def main():

    st.title("ML Обработка Видео")
    st.markdown("Загрузите видео для обнаружения и распознавания товаров с помощью ML")
    user_id = st.text_input("User ID", value="anonymous", help="Используется для списка прошлых обработок")
    callback_url = st.text_input(
        "Callback URL",
        value="",
        help="Опционально: webhook, куда backend отправит POST после завершения задачи",
    )
    display_previous_jobs(user_id)
    display_run_comparison(user_id)

    render_drag_and_drop_hint()
    
    uploaded_file = st.file_uploader(
        "Перетащите видео сюда или выберите файл",
        type=["mp4", "mov", "avi"],
        help="""-Обнаружение товаров в кадрах видео

-Распознавание названий товаров

-Извлечение штрих-кодов и цен

-Генерация CSV результатов"""
    )
    
    if uploaded_file is not None:
        is_too_large = display_uploaded_video_info(uploaded_file)
        if st.button("Начать обработку", type="primary", disabled=is_too_large):
            process_video(uploaded_file, user_id, callback_url)


def render_drag_and_drop_hint():
    st.markdown(
        """
        <div style="
            border: 2px dashed #6aa6ff;
            border-radius: 14px;
            padding: 18px;
            margin: 12px 0;
            background: rgba(106, 166, 255, 0.08);
            text-align: center;
        ">
            <b>Drag-and-drop зона</b><br/>
            Перетащите .mp4, .mov или .avi файл в поле ниже. Максимальный размер: 500 MB.
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_uploaded_video_info(uploaded_file) -> bool:
    size_bytes = uploaded_file.getbuffer().nbytes
    st.info(f"Файл: `{uploaded_file.name}` | Размер: {format_file_size(size_bytes)}")

    is_too_large = size_bytes > MAX_FILE_SIZE_BYTES
    if is_too_large:
        st.warning(f"Размер файла превышает лимит {MAX_FILE_SIZE_MB} MB. Загрузка на обработку отключена.")

    frame = extract_first_frame(uploaded_file)
    if frame is not None:
        st.subheader("Предпросмотр первого кадра")
        st.image(frame, channels="RGB", use_column_width=True)
    else:
        st.warning("Не удалось извлечь первый кадр для предпросмотра")

    return is_too_large


def format_file_size(size_bytes: int) -> str:
    size_mb = size_bytes / 1024 / 1024
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def extract_first_frame(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def process_video(uploaded_file, user_id: str, callback_url: str = ""):
    try:
        with st.spinner("Загрузка видео..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"user_id": user_id or "anonymous"}
            if callback_url.strip():
                data["callback_url"] = callback_url.strip()
            
            response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=30)
            
            if response.status_code != 200:
                st.error(f"Ошибка загрузки: {response.json().get('detail', 'Неизвестная ошибка')}")
                return
            
            job_id = response.json()["job_id"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            try:
                response = requests.get(f"{API_URL}/status/{job_id}", timeout=10)
                
                if response.status_code != 200:
                    st.error("Не удалось получить статус")
                    break
                
                status_data = response.json()
                status = status_data["status"]
                progress = status_data["progress"]
                message = status_data["message"]
                
                progress_bar.progress(progress / 100)
                status_text.text(f"Статус: {status.upper()} - {message}")
                
                if status == "done":
                    display_results(job_id)
                    break
                elif status == "error":
                    st.error(f"Ошибка обработки: {message}")
                    break

                time.sleep(1.5)
                
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                # transient: API занят тяжёлой обработкой, retry без падения
                status_text.text(f"Статус: соединение... ({e.__class__.__name__})")
                time.sleep(3)
                continue
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения: {str(e)}")
                break
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")


def display_results(job_id: str):
    st.header("Результаты распознавания")

    try:
        csv_response = requests.get(f"{API_URL}/result/{job_id}.csv", timeout=10)
        if csv_response.status_code != 200:
            st.error("Не удалось загрузить CSV с результатами")
            return

        df = pd.read_csv(io.BytesIO(csv_response.content))
    except Exception as e:
        st.error(f"Не удалось загрузить данные: {str(e)}")
        return

    preview_col, download_col = st.columns([2, 1])
    with preview_col:
        display_detection_preview(job_id)
    with download_col:
        render_result_downloads(job_id, csv_response.content)
        problem_count = int(build_problem_mask(df).sum())
        if problem_count:
            st.warning(f"Проблемные строки: {problem_count}")
        else:
            st.success("Проблемных строк не найдено")

    filtered_df = display_interactive_table(df)
    display_detection_map(job_id, filtered_df)


def render_result_downloads(job_id: str, csv_content: bytes):
    st.subheader("Экспорт")
    st.download_button(
        label="CSV",
        data=csv_content,
        file_name=f"results_{job_id}.csv",
        mime="text/csv",
        type="primary",
    )

    export_formats = [
        ("XLSX для Excel", "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("JSON", "json", "application/json"),
        ("Parquet", "parquet", "application/octet-stream"),
    ]

    for label, extension, mime in export_formats:
        content = load_export_bytes(job_id, extension)
        if content is None:
            st.caption(f"{label}: недоступен")
            continue

        st.download_button(
            label=label,
            data=content,
            file_name=f"results_{job_id}.{extension}",
            mime=mime,
            key=f"download-{extension}-{job_id}",
        )


@st.cache_data(show_spinner=False)
def load_export_bytes(job_id: str, extension: str):
    try:
        response = requests.get(f"{API_URL}/result/{job_id}.{extension}", timeout=30)
        if response.status_code == 200:
            return response.content
    except requests.exceptions.RequestException:
        return None
    return None


def display_detection_preview(job_id: str):
    st.subheader("Общий предпросмотр")
    try:
        preview_url = f"{API_URL}/preview/{job_id}.jpg"
        response = requests.get(preview_url, timeout=10)

        if response.status_code == 200:
            image = Image.open(requests.get(preview_url, stream=True, timeout=10).raw)
            st.image(image)
        else:
            st.warning("Изображение предпросмотра недоступно")
    except Exception as e:
        st.error(f"Не удалось загрузить предпросмотр: {str(e)}")


def display_interactive_table(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Таблица результатов")
    st.caption("Фильтры работают по подстроке. Сортировка доступна кликом по заголовкам таблицы.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        name_query = st.text_input("Поиск по product_name", key="filter-product-name")
    with filter_col2:
        barcode_query = st.text_input("Поиск по barcode", key="filter-barcode")
    with filter_col3:
        price_query = st.text_input("Поиск по цене", key="filter-price")

    filtered_df = apply_result_filters(df, name_query, barcode_query, price_query)
    filtered_df = filtered_df.copy()
    filtered_df.insert(0, "problem", build_problem_mask(filtered_df))

    st.dataframe(
        filtered_df.style.apply(highlight_problem_rows, axis=1),
        height=420,
        use_container_width=True,
    )
    st.caption(f"Показано {len(filtered_df)} из {len(df)} строк. Красные строки: пустые обязательные поля.")
    return filtered_df.drop(columns=["problem"])


def apply_result_filters(
    df: pd.DataFrame,
    name_query: str,
    barcode_query: str,
    price_query: str,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if name_query and "product_name" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["product_name"].astype(str).str.contains(name_query, case=False, na=False)
        ]

    if barcode_query and "barcode" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["barcode"].astype(str).str.contains(barcode_query, case=False, na=False)
        ]

    if price_query:
        existing_price_cols = [col for col in PRICE_COLUMNS if col in filtered_df.columns]
        if existing_price_cols:
            price_mask = pd.Series(False, index=filtered_df.index)
            for col in existing_price_cols:
                price_mask |= filtered_df[col].astype(str).str.contains(price_query, case=False, na=False)
            filtered_df = filtered_df[price_mask]

    return filtered_df


def build_problem_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    for col in MANDATORY_COLUMNS:
        if col in df.columns:
            mask |= df[col].apply(is_empty_value)

    existing_price_cols = [col for col in PRICE_COLUMNS if col in df.columns]
    if existing_price_cols:
        has_price = pd.Series(False, index=df.index)
        for col in existing_price_cols:
            has_price |= ~df[col].apply(is_empty_value)
        mask |= ~has_price

    return mask


def is_empty_value(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def highlight_problem_rows(row):
    if bool(row.get("problem", False)):
        return ["background-color: #ffe0e0"] * len(row)
    return [""] * len(row)


def display_detection_map(job_id: str, df: pd.DataFrame):
    st.subheader("Карта детекций")

    if df.empty:
        st.info("Нет строк для отображения на видео после фильтрации")
        return

    required = {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
    if not required.issubset(df.columns):
        st.warning("В CSV нет координат или frame_timestamp для карты детекций")
        return

    video_bytes = load_video_bytes(job_id)
    if not video_bytes:
        st.warning("Исходное видео недоступно. Возможно, файлы уже удалены автоочисткой.")
        return

    options = build_detection_options(df)
    selected_label = st.selectbox("Выберите ценник", options=list(options.keys()))
    selected_row = df.loc[options[selected_label]]
    timestamp_ms = safe_int(selected_row.get("frame_timestamp"), 0)
    start_time = max(timestamp_ms // 1000 - 1, 0)

    video_col, frame_col = st.columns(2)
    with video_col:
        st.caption(f"Видео открыто около {timestamp_ms} ms")
        st.video(video_bytes, start_time=start_time)

    with frame_col:
        frame = extract_frame_with_bbox(video_bytes, selected_row, timestamp_ms)
        if frame is not None:
            st.caption("Кадр с bbox выбранного ценника")
            st.image(frame, channels="RGB", use_column_width=True)
        else:
            st.warning("Не удалось отрисовать bbox на кадре")


def build_detection_options(df: pd.DataFrame) -> dict:
    options = {}
    for index, row in df.iterrows():
        product_name = str(row.get("product_name", "")).strip() or "без названия"
        barcode = str(row.get("barcode", "")).strip() or "без barcode"
        timestamp = safe_int(row.get("frame_timestamp"), 0)
        price = first_non_empty(row, PRICE_COLUMNS) or "без цены"
        label = f"{index}: {product_name} | {barcode} | {price} | {timestamp} ms"
        options[label] = index
    return options


def first_non_empty(row, columns):
    for col in columns:
        if col in row and not is_empty_value(row[col]):
            return str(row[col])
    return ""


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False)
def load_video_bytes(job_id: str):
    try:
        response = requests.get(f"{API_URL}/video/{job_id}.mp4", timeout=30)
        if response.status_code == 200:
            return response.content
    except requests.exceptions.RequestException:
        return None
    return None


def extract_frame_with_bbox(video_bytes: bytes, row: pd.Series, timestamp_ms: int):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        x_min = safe_int(row.get("x_min"))
        y_min = safe_int(row.get("y_min"))
        x_max = safe_int(row.get("x_max"))
        y_max = safe_int(row.get("y_max"))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        label = first_non_empty(row, ["product_name", "barcode"])
        if label:
            cv2.putText(
                frame,
                label[:40],
                (x_min, max(25, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def display_previous_jobs(user_id: str):
    with st.expander("Мои предыдущие обработки", expanded=False):
        try:
            jobs = fetch_user_jobs(user_id, limit=20)
            if not jobs:
                st.caption("История пуста")
                return

            for job in jobs:
                job_id = job["id"]
                st.write(
                    f"`{job_id}` | {job['status']} | {job['progress']}% | {job['created_at']}"
                )
                if job["status"] == "done" and job.get("csv_path"):
                    st.link_button(
                        "Скачать CSV",
                        f"{API_URL}/result/{job_id}.csv",
                    )
        except requests.exceptions.RequestException:
            st.warning("Не удалось загрузить историю обработок")


def display_run_comparison(user_id: str):
    with st.expander("Сравнение нескольких прогонов", expanded=False):
        try:
            jobs = [
                job
                for job in fetch_user_jobs(user_id, limit=100)
                if job["status"] == "done" and job.get("csv_path")
            ]
        except requests.exceptions.RequestException:
            st.warning("Не удалось загрузить список прогонов")
            return

        if len(jobs) < 2:
            st.caption("Для сравнения нужно минимум два завершённых прогона с доступными CSV.")
            return

        labels = {format_job_label(job): job["id"] for job in jobs}
        left_col, right_col = st.columns(2)

        with left_col:
            left_label = st.selectbox("Первый прогон", list(labels.keys()), key="compare-left")
        with right_col:
            right_options = [label for label in labels if label != left_label]
            right_label = st.selectbox("Второй прогон", right_options, key="compare-right")

        if st.button("Сравнить CSV", key="compare-runs"):
            left_job_id = labels[left_label]
            right_job_id = labels[right_label]
            left_df = load_result_csv(left_job_id)
            right_df = load_result_csv(right_job_id)

            if left_df is None or right_df is None:
                st.error("Не удалось загрузить один из CSV для сравнения")
                return

            comparison = build_runs_comparison(left_df, right_df)
            render_runs_comparison(comparison, left_job_id, right_job_id)


def fetch_user_jobs(user_id: str, limit: int = 50):
    response = requests.get(
        f"{API_URL}/jobs",
        params={"user_id": user_id or "anonymous", "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("jobs", [])


def format_job_label(job: dict) -> str:
    return f"{job['created_at']} | {job['id'][:8]} | {job['progress']}%"


@st.cache_data(show_spinner=False)
def load_result_csv(job_id: str):
    try:
        response = requests.get(f"{API_URL}/result/{job_id}.csv", timeout=20)
        if response.status_code != 200:
            return None
        return pd.read_csv(io.BytesIO(response.content))
    except Exception:
        return None


def build_runs_comparison(left_df: pd.DataFrame, right_df: pd.DataFrame) -> dict:
    left_norm = normalize_for_comparison(left_df)
    right_norm = normalize_for_comparison(right_df)

    if left_norm.empty or right_norm.empty:
        return {
            "summary": {
                "left": len(left_norm),
                "right": len(right_norm),
                "common": 0,
                "price_diff": 0,
                "left_only": len(left_norm),
                "right_only": len(right_norm),
            },
            "common": pd.DataFrame(),
            "price_diff": pd.DataFrame(),
            "left_only": left_norm,
            "right_only": right_norm,
        }

    common = left_norm.merge(
        right_norm,
        on="compare_key",
        how="inner",
        suffixes=("_left", "_right"),
    )
    price_diff = common[build_price_diff_mask(common)].copy()
    left_only = left_norm[~left_norm["compare_key"].isin(right_norm["compare_key"])].copy()
    right_only = right_norm[~right_norm["compare_key"].isin(left_norm["compare_key"])].copy()

    return {
        "summary": {
            "left": len(left_norm),
            "right": len(right_norm),
            "common": len(common),
            "price_diff": len(price_diff),
            "left_only": len(left_only),
            "right_only": len(right_only),
        },
        "common": common,
        "price_diff": price_diff,
        "left_only": left_only,
        "right_only": right_only,
    }


def normalize_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["compare_key"] = normalized.apply(build_compare_key, axis=1)
    normalized = normalized[normalized["compare_key"] != ""].copy()
    normalized = normalized.drop_duplicates(subset=["compare_key"], keep="first")

    columns = ["compare_key"]
    for col in ["id_sku", "barcode", "product_name", *PRICE_COLUMNS]:
        if col in normalized.columns:
            columns.append(col)

    return normalized[columns]


def build_compare_key(row: pd.Series) -> str:
    for col in COMPARE_KEY_COLUMNS:
        if col in row and not is_empty_value(row[col]) and str(row[col]).strip().lower() != "нет":
            return f"{col}:{str(row[col]).strip()}"
    return ""


def build_price_diff_mask(comparison_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=comparison_df.index)
    for col in PRICE_COLUMNS:
        left_col = f"{col}_left"
        right_col = f"{col}_right"
        if left_col in comparison_df.columns and right_col in comparison_df.columns:
            mask |= (
                comparison_df[left_col].apply(normalize_price_value)
                != comparison_df[right_col].apply(normalize_price_value)
            )
    return mask


def normalize_price_value(value) -> str:
    if is_empty_value(value):
        return ""
    return str(value).strip().replace(",", ".").replace(" ", "")


def render_runs_comparison(comparison: dict, left_job_id: str, right_job_id: str):
    summary = comparison["summary"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Общие SKU/barcode", summary["common"])
    metric_cols[1].metric("Расхождения цен", summary["price_diff"])
    metric_cols[2].metric("Только в первом", summary["left_only"])
    metric_cols[3].metric("Только во втором", summary["right_only"])

    st.caption(f"Первый: `{left_job_id}` | Второй: `{right_job_id}`")

    tab_common, tab_diff, tab_left_only, tab_right_only = st.tabs(
        ["Side-by-side", "Расхождения цен", "Только первый", "Только второй"]
    )

    with tab_common:
        st.dataframe(comparison["common"], use_container_width=True, height=360)

    with tab_diff:
        if comparison["price_diff"].empty:
            st.success("Расхождений цен среди общих SKU/barcode не найдено")
        else:
            st.dataframe(
                comparison["price_diff"].style.apply(highlight_price_diff_rows, axis=1),
                use_container_width=True,
                height=360,
            )

    with tab_left_only:
        st.dataframe(comparison["left_only"], use_container_width=True, height=320)

    with tab_right_only:
        st.dataframe(comparison["right_only"], use_container_width=True, height=320)


def highlight_price_diff_rows(row):
    styles = []
    for col in row.index:
        base = col.removesuffix("_left").removesuffix("_right")
        if base in PRICE_COLUMNS:
            left_value = row.get(f"{base}_left", "")
            right_value = row.get(f"{base}_right", "")
            if normalize_price_value(left_value) != normalize_price_value(right_value):
                styles.append("background-color: #fff0b3")
                continue
        styles.append("")
    return styles


if __name__ == "__main__":
    main()
