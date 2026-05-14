import time
import os
from pathlib import Path
import requests
import streamlit as st
import pandas as pd
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ML Обработка Видео",
    layout="wide"
)


def main():

    st.title("ML Обработка Видео")
    st.markdown("Загрузите видео для обнаружения и распознавания товаров с помощью ML")
    
    uploaded_file = st.file_uploader(
        "Выберите видео файл",
        type=["mp4", "mov", "avi"],
        help="""-Обнаружение товаров в кадрах видео

-Распознавание названий товаров

-Извлечение штрих-кодов и цен

-Генерация CSV результатов"""
    )
    
    if uploaded_file is not None:
        if st.button("Начать обработку", type="primary"):
            process_video(uploaded_file)


def process_video(uploaded_file):
    try:
        with st.spinner("Загрузка видео..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            
            response = requests.post(f"{API_URL}/upload", files=files, timeout=30)
            
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
                
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения: {str(e)}")
                break
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")


def display_results(job_id: str):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Предпросмотр обнаружений")
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
    
    with col2:
        st.subheader("Предпросмотр данных")
        try:
            csv_url = f"{API_URL}/result/{job_id}.csv"
            df = pd.read_csv(csv_url)
            
            st.dataframe(df.head(20), height=400)
            st.caption(f"Показаны первые 20 из {len(df)} обнаружений")
            
        except Exception as e:
            st.error(f"Не удалось загрузить данные: {str(e)}")
    
    try:
        csv_url = f"{API_URL}/result/{job_id}.csv"
        response = requests.get(csv_url, timeout=10)
        
        if response.status_code == 200:
            st.download_button(
                label="Скачать полные результаты CSV",
                data=response.content,
                file_name=f"results_{job_id}.csv",
                mime="text/csv",
                type="primary"
            )
    except Exception as e:
        st.error(f"Не удалось подготовить загрузку: {str(e)}")


if __name__ == "__main__":
    main()
