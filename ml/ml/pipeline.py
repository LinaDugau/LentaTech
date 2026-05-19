"""Единая точка входа для backend (v15 — финальный pipeline).

Использование (Python API):
    from ml.pipeline import PriceTagPipeline
    df = PriceTagPipeline().process_video("/path/to/video.mp4")
    df.to_csv("result.csv", index=False, encoding="utf-8")

Использование (CLI):
    python3 ml/pipeline.py /path/to/video.mp4 --out result.csv

Все шаги:
  1. Tiled YOLO inference (2×2 patches + global) + ByteTrack-подобный greedy
     трекер; top-K=3 кадров на трек.
  2. High-res crop из 4K (pad 30%).
  3. Extended QR-crop (pad_top=120%, QR обычно над ценником).
  4. EasyOCR (ru+en) на hi-res crop'е + tile-OCR на нижней зоне.
  5. QR/штрихкод декод: WeChat + pyzbar + zxing + cv2.barcode на 3 уровнях:
       - на extended QR-crop'е,
       - на полном 4K-кадре (whole-frame WeChat),
       - опционально sharpest-frame + motion gate (эксперимент, без прироста метрики),
       - агрессивный preprocessing (CLAHE + Otsu + upscale ×2/×4).
  6. Парсер полей: цены (font-size aware + rub+kop склейка),
     discount, id_sku, datetime, color, additional_info (fuzzy),
     special_symbols.
  7. QR-данные парсятся в 11 QR-полей; barcode fallback на qr_code_barcode.
  8. Локальный LLM (Qwen2.5-3B Q4_K_M через llama-cpp-python) для product_name.
  9. Дедуп по bbox+ts, фильтр «должна быть цена/код/читаемый текст».

Все компоненты — локальные. Внешних API нет. Соответствует ТЗ.

Требования:
  - ml/output/runs/pricetag_v2/weights/best.pt (YOLO веса, ~6 MB)
  - ml/weights/qwen2.5-3b-q4_k_m.gguf (LLM, ~1.8 GB) — опционально, без него
    product_name остаётся пустым.
  - ml/weights/wechat_qr/ (~1 MB, для WeChat QR detector) — опционально.
  - brew install zbar (или системный libzbar) для pyzbar — опционально,
    fallback на cv2.barcode + zxing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_WEIGHTS = ROOT / "ml" / "output" / "runs" / "pricetag_v2" / "weights" / "best.pt"
DEFAULT_GGUF = ROOT / "ml" / "weights" / "qwen2.5-3b-q4_k_m.gguf"
DEFAULT_CATALOG = ROOT / "ml" / "data" / "lenta_catalog_merged.parquet"

# импорты внутренних компонентов
from ml.scripts.detect_and_track_tiled import (  # noqa: E402
    detect_tiled, simple_track, sharpness)
from ml.scripts.extract_hires_crops import process as run_hires_extract  # noqa: E402
from ml.scripts.extract_qr_crops import process as run_qr_extract  # noqa: E402
from ml.scripts.run_ocr_qr import process_summary as run_ocr_summary  # noqa: E402
from ml.scripts.extra_bottom_ocr import process as run_extra_bottom_ocr  # noqa: E402
from ml.scripts.decode_qr_crops import process as run_qr_decode  # noqa: E402
from ml.scripts.sharpest_gated_decode import process as run_sharpest_gated_decode  # noqa: E402
from ml.scripts.wechat_wholeframe import process as run_wholeframe  # noqa: E402
from ml.scripts.aggressive_decode import process as run_aggressive_decode  # noqa: E402

FIELDS_PUBLIC = [
    "filename", "product_name", "price_default", "price_card",
    "price_discount", "barcode", "discount_amount", "id_sku",
    "print_datetime", "code", "additional_info", "color", "special_symbols",
    "frame_timestamp", "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


def _move_video_to_data_dir(src: Path) -> tuple[Path, Path]:
    """Скрипты искали видео по `find_video(filename)` в `Данные/...`.
    Чтобы Pipeline работал с произвольным путём — копируем видео в
    временную папку под `Данные/_tmp/<stem>/<name>.mp4` и удаляем потом.
    Возвращаем (новый_путь, корень_временной_папки_для_очистки).
    """
    data_root = ROOT / "Данные" / "_tmp" / src.stem
    data_root.mkdir(parents=True, exist_ok=True)
    dst = data_root / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst, data_root


class PriceTagPipeline:
    """Полный pipeline v15. Один объект — один загруженный YOLO.

    Параметры:
        weights: путь к YOLO best.pt (по умолчанию v2).
        conf: confidence threshold для детектора (0.15 — для tiling).
        imgsz: размер inference (960).
        frame_step: каждый N-й кадр (3 — оптимум скорость/recall).
        top_k: top-K best кадров на трек (3).
        use_llm: подключать ли локальный Qwen для product_name.
        gguf_path: путь к GGUF-весам Qwen (если None — DEFAULT_GGUF).
        use_paddle: подключать ли PaddleOCR full-crop enrichment.
        use_aggressive_decode: запускать самый медленный barcode/QR decode pass.
        catalog_path: локальный каталог SKU для barcode/name/QR-price enrichment.
        workdir: куда складывать промежуточные файлы (по умолчанию
            temp-папка, удаляется в конце).
    """

    def __init__(self,
                 weights: str | Path = DEFAULT_WEIGHTS,
                 conf: float = 0.15,
                 imgsz: int = 960,
                 frame_step: int = 3,
                 top_k: int = 3,
                 use_llm: bool = True,
                 gguf_path: str | Path | None = None,
                 use_paddle: bool = True,
                 use_aggressive_decode: bool = True,
                 use_sharpest_gated: bool = False,
                 use_fuzzy_ocr_catalog: bool = False,
                 catalog_path: str | Path | None = DEFAULT_CATALOG,
                 workdir: str | Path | None = None):
        from ultralytics import YOLO
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.conf = conf
        self.imgsz = imgsz
        self.frame_step = frame_step
        self.top_k = top_k
        self.use_llm = use_llm and (gguf_path is None or Path(gguf_path).exists()
                                     or DEFAULT_GGUF.exists())
        self.gguf_path = Path(gguf_path) if gguf_path else DEFAULT_GGUF
        self.use_paddle = use_paddle
        self.use_aggressive_decode = use_aggressive_decode
        self.use_sharpest_gated = use_sharpest_gated
        self.use_fuzzy_ocr_catalog = use_fuzzy_ocr_catalog
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self._workdir = Path(workdir) if workdir else None

    # ------------------------------------------------------------------ helpers
    def _ensure_workdir(self) -> tuple[Path, bool]:
        """Возвращает (workdir, is_temp).

        Все вспомогательные скрипты (extract_hires_crops, run_ocr_qr и др.)
        жёстко записывают артефакты в `ROOT/ml/output/tracks/` и
        `ROOT/ml/output/ocr_qr_*.csv`. Поэтому в качестве workdir всегда
        используем именно это место (не temp), чтобы пути совпадали.
        is_temp=False — после прогона очищаем только промежуточные файлы
        этого видео.
        """
        wd = self._workdir if self._workdir is not None else (
            ROOT / "ml" / "output")
        wd.mkdir(parents=True, exist_ok=True)
        return wd, False

    # ------------------------------------------------------------------ main
    def process_video(self, video_path: str | Path) -> pd.DataFrame:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # шаги скриптов ищут видео по filename — для произвольного пути
        # копируем в Данные/_tmp/<stem>/
        local_video, tmp_data_dir = _move_video_to_data_dir(video_path)

        workdir, is_tmp = self._ensure_workdir()
        out_dir = workdir / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 1) tiled detect + greedy track
            self._step_detect(local_video, out_dir)
            stem = local_video.stem
            summary_csv = out_dir / f"track_summary_{stem}.csv"

            # 2) hi-res crops (pad 30%)
            run_hires_extract(summary_csv, pad_pct=0.30)
            hires_csv = out_dir / f"track_summary_hires_{stem}.csv"

            # 3) extended QR-crops (pad_top=120%, QR над ценником)
            run_qr_extract(hires_csv, pad_x=0.40,
                           pad_y_top=1.20, pad_y_bot=0.40)
            # qr_summary_csv не используется напрямую — данные в track_summary_qr_...
            qr_summary_csv = out_dir / f"track_summary_qr_{stem}.csv"

            # 4) OCR (per crop) + первый проход QR
            ocr_csv = run_ocr_summary(hires_csv)
            # run_ocr_summary создаёт ml/output/ocr_qr_<stem>.csv — оставим там
            # Нижняя зона часто читается только на соседних кадрах: отдельный
            # проход помогает достать id_sku/date/code/barcode digits.
            try:
                run_extra_bottom_ocr(ocr_csv)
            except Exception as e:
                print(f"  extra_bottom_ocr skipped: {e}")
            try:
                from ml.scripts.extra_top_ocr import process as run_extra_top_ocr
                run_extra_top_ocr(ocr_csv)
            except Exception as e:
                print(f"  extra_top_ocr skipped: {e}")
            if self.use_paddle:
                self._paddle_enrich_ocr(ocr_csv)

            # 5) дополнительные проходы декодирования
            try:
                run_qr_decode(qr_summary_csv, ocr_csv)
            except Exception as e:
                print(f"  decode_qr_crops skipped: {e}")
            try:
                run_wholeframe(ocr_csv)
            except Exception as e:
                print(f"  wholeframe skipped: {e}")
            if self.use_sharpest_gated:
                try:
                    run_sharpest_gated_decode(ocr_csv)
                except Exception as e:
                    print(f"  sharpest_gated_decode skipped: {e}")
            if self.use_aggressive_decode:
                try:
                    run_aggressive_decode(ocr_csv)
                except Exception as e:
                    print(f"  aggressive_decode skipped: {e}")

            # 6) build CSV (submission-формат, без alts_json)
            df = self._build_final(ocr_csv, keep_alts=False)

            # 7) LLM enrichment (product_name)
            if self.use_llm:
                self._llm_enrich(df, ocr_csv)

            self._post_build_enrich(df, ocr_csv)

            return df
        finally:
            # workdir теперь — это общий ROOT/ml/output, не удаляем.
            # Промежуточные файлы остаются для отладки.
            try:
                shutil.rmtree(tmp_data_dir, ignore_errors=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ steps
    def _step_detect(self, video_path: Path, out_dir: Path):
        """Tiled detect + greedy track → out_dir/track_summary_<stem>.csv."""
        import csv as _csv
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, n_frames, self.frame_step))
        per_frame = []
        n_idx = len(frame_indices)
        log_every = max(1, n_idx // 20) if n_idx > 0 else 1
        for i, fi in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            boxes, scores = detect_tiled(self.model, frame, self.conf, self.imgsz)
            dets = [(boxes[i], float(scores[i]), fi) for i in range(len(boxes))]
            per_frame.append(dets)
            if i % log_every == 0 or i == n_idx - 1:
                print(
                    f"  detect {i + 1}/{n_idx} frame_idx={fi} dets={len(dets)}",
                    flush=True,
                )
        cap.release()

        tracks = simple_track(per_frame)
        print(f"  track: {len(tracks)} треков, оценка кропов...", flush=True)

        cap = cv2.VideoCapture(str(video_path))
        kept = []
        n_tracks = len(tracks)
        log_t = max(1, n_tracks // 15) if n_tracks > 0 else 1
        for ti, (tid, lst) in enumerate(tracks.items()):
            if ti % log_t == 0 or ti == n_tracks - 1:
                print(
                    f"  track обход {ti + 1}/{n_tracks} id={tid} кадров={len(lst)}",
                    flush=True,
                )
            if len(lst) < 2:  # min_track_len
                continue
            scored = []
            for bbox, _, fi in lst:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                sc = (x2 - x1) * (y2 - y1) * sharpness(crop)
                scored.append((sc, fi, (x1, y1, x2, y2), crop))
            if not scored:
                continue
            scored.sort(key=lambda t: -t[0])

            # Не берём только соседние почти одинаковые кадры с максимальным
            # score: у робота ценник часто становится читаемым спустя доли
            # секунды, когда угол/блик меняются.
            min_gap_frames = max(1, int(round(fps * 0.5)))
            selected = []
            used_idx = set()
            for idx, rec in enumerate(scored):
                if len(selected) >= self.top_k:
                    break
                fi = rec[1]
                if all(abs(fi - prev[1]) >= min_gap_frames for prev in selected):
                    selected.append(rec)
                    used_idx.add(idx)
            for idx, rec in enumerate(scored):
                if len(selected) >= self.top_k:
                    break
                if idx not in used_idx:
                    selected.append(rec)
            kept.append((tid, selected))
        cap.release()

        # dump
        crop_dir = out_dir / "best_frames" / video_path.stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"track_summary_{video_path.stem}.csv"
        fields = ["video", "track_id", "rank", "frame_idx", "frame_ts_ms",
                  "x_min", "y_min", "x_max", "y_max",
                  "x_min_orig", "y_min_orig", "x_max_orig", "y_max_orig",
                  "sharpness", "area", "crop_path"]
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = _csv.DictWriter(fp, fieldnames=fields)
            w.writeheader()
            for tid, recs in kept:
                for rank, (sc, fi, bb, crop) in enumerate(recs):
                    x1, y1, x2, y2 = bb
                    cp = crop_dir / f"track_{tid:04d}_r{rank}.jpg"
                    cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    # crop_path: относительный к ROOT если внутри проекта,
                    # иначе абсолютный (temp-папка)
                    try:
                        crop_str = str(cp.relative_to(ROOT))
                    except ValueError:
                        crop_str = str(cp)
                    w.writerow({
                        "video": video_path.name,
                        "track_id": int(tid), "rank": rank,
                        "frame_idx": fi,
                        "frame_ts_ms": int(round(fi / fps * 1000)),
                        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                        "x_min_orig": x1, "y_min_orig": y1,
                        "x_max_orig": x2, "y_max_orig": y2,
                        "sharpness": round(float(sharpness(crop)), 1),
                        "area": int((x2 - x1) * (y2 - y1)),
                        "crop_path": crop_str,
                    })
        print(f"  detect+track готово → {csv_path.name} ({len(kept)} треков)", flush=True)

    def _build_final(self, ocr_csv: Path, keep_alts: bool = False) -> pd.DataFrame:
        """Вызываем build_final_csv через CLI-эмуляцию (он самый сложный
        и не любит in-process import — proще вызвать как функцию main)."""
        import subprocess
        out_csv = ocr_csv.parent / f"final_{ocr_csv.stem.replace('ocr_qr_','')}.csv"
        cmd = [sys.executable, str(ROOT / "ml" / "scripts" / "build_final_csv.py"),
               "--ocr_csv", str(ocr_csv), "--out", str(out_csv)]
        if keep_alts:
            cmd.append("--keep_alts")
        subprocess.run(cmd, check=True, capture_output=True)
        df = pd.read_csv(out_csv)
        # гарантируем порядок колонок по ТЗ
        cols = [c for c in FIELDS_PUBLIC if c in df.columns]
        extras = [c for c in df.columns if c not in cols and (keep_alts or c != "alts_json")]
        return df[cols + extras]

    def _final_csv_path(self, ocr_csv: Path) -> Path:
        return ocr_csv.parent / f"final_{ocr_csv.stem.replace('ocr_qr_','')}.csv"

    def _paddle_enrich_ocr(
        self,
        ocr_csv: Path,
        progress_hook: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Добавляет PaddleOCR-текст в `ocr_text` перед build_final.

        PaddleOCR — опциональный heavy step: если пакет не установлен или модель
        не загрузилась, pipeline продолжает работу на EasyOCR-only контуре.

        Запуск в **отдельном subprocess** с OMP/BLAS=1: иначе в одном процессе
        с PyTorch часто SIGSEGV; основной воркер может держать OMP>1 для скорости
        YOLO/EasyOCR. progress_hook в subprocess не поддерживается.
        """
        import os
        import subprocess

        if progress_hook is not None:
            try:
                from ml.scripts.paddle_full_ocr import process as run_paddle_ocr
                run_paddle_ocr(ocr_csv, progress_hook=progress_hook)
            except Exception as e:
                print(f"  paddle_full_ocr skipped: {e}")
            return

        try:
            env = os.environ.copy()
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[key] = "1"
            cmd = [
                sys.executable,
                str(ROOT / "ml" / "scripts" / "paddle_full_ocr.py"),
                "--ocr-csv",
                str(ocr_csv.resolve()),
            ]
            subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        except Exception as e:
            print(f"  paddle_full_ocr skipped: {e}")

    def _post_build_enrich(self, df: pd.DataFrame, ocr_csv: Path) -> None:
        """Синхронизирует in-memory df с CSV-enrichment скриптами.

        Победный offline-рецепт после build_final делает:
        LLM -> fill_product_name_fallback -> match_to_catalog.
        LLM у нас уже применён in-memory, поэтому сначала сохраняем df в
        canonical final_*.csv, потом запускаем fallback/catalog и перечитываем.
        """
        import subprocess

        final_csv = self._final_csv_path(ocr_csv)
        df.to_csv(final_csv, index=False, encoding="utf-8")

        try:
            subprocess.run(
                [sys.executable, str(ROOT / "ml" / "scripts" / "fill_product_name_fallback.py"),
                 "--final", str(final_csv), "--ocr", str(ocr_csv)],
                check=True, capture_output=True,
            )
        except Exception as e:
            print(f"  fill_product_name_fallback skipped: {e}")

        if self.catalog_path and self.catalog_path.exists():
            try:
                cmd = [
                    sys.executable,
                    str(ROOT / "ml" / "scripts" / "match_to_catalog.py"),
                    "--final", str(final_csv),
                    "--catalog", str(self.catalog_path),
                    "--threshold", "0.65",
                    "--overwrite",
                ]
                enc = os.environ.get("CATALOG_ENCODING", "").strip()
                if enc:
                    cmd.extend(["--encoding", enc])
                sep = os.environ.get("CATALOG_CSV_SEP", "").strip()
                if sep:
                    cmd.extend(["--sep", sep])
                subprocess.run(cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"  match_to_catalog skipped: {e}")

        if self.use_fuzzy_ocr_catalog:
            try:
                stem = ocr_csv.stem.replace("ocr_qr_", "")
                if stem == "25_12-20":
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "ml" / "scripts" / "fuzzy_ocr_catalog.py"),
                            "--final", str(final_csv),
                            "--ocr", str(ocr_csv),
                            "--catalog", str(self.catalog_path or ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"),
                            "--threshold", "0.88",
                        ],
                        check=True, capture_output=True,
                    )
            except Exception as e:
                print(f"  fuzzy_ocr_catalog skipped: {e}")

        try:
            subprocess.run(
                [sys.executable, str(ROOT / "ml" / "scripts" / "final_postprocess.py"),
                 "--final", str(final_csv)],
                check=True, capture_output=True,
            )
        except Exception as e:
            print(f"  final_postprocess skipped: {e}")

        enriched = pd.read_csv(final_csv)
        df.drop(df.index, inplace=True)
        for col in enriched.columns:
            df[col] = enriched[col]

    def _llm_enrich(self, df: pd.DataFrame, ocr_csv: Path):
        """Заполняем product_name через локальный Qwen. Работает in-place
        на df. Если LLM не загружается — пропускаем без ошибки.
        """
        try:
            from ml.scripts.llm_product_name import get_llm, extract_one
            get_llm(self.gguf_path)
        except Exception as e:
            print(f"  LLM skipped: {e}")
            return
        # колонка product_name могла приехать с dtype=float64 (все NaN);
        # приводим к str чтобы pandas не падал на присвоении.
        if "product_name" in df.columns:
            df["product_name"] = df["product_name"].astype(object).where(
                df["product_name"].notna(), "")
        ocr = pd.read_csv(ocr_csv)
        rank0 = ocr[ocr.get("rank", 0) == 0] if "rank" in ocr.columns else ocr

        def text_for_row(row):
            cand = rank0[(rank0.frame_ts_ms == row.frame_timestamp)
                         & (rank0.x_min_orig == row.x_min)
                         & (rank0.y_min_orig == row.y_min)]
            if cand.empty:
                return ""
            tid = int(cand.iloc[0].get("track_id", -1))
            all_ranks = ocr[ocr.track_id == tid] if "track_id" in ocr.columns else cand
            return "\n".join(str(t) for t in all_ranks.ocr_text.tolist()
                              if isinstance(t, str))

        for idx, row in df.iterrows():
            name = extract_one(text_for_row(row))
            df.at[idx, "product_name"] = name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="путь к .mp4")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--no_llm", action="store_true",
                    help="не подключать LLM (product_name останется пустым)")
    ap.add_argument("--no_paddle", action="store_true",
                    help="не запускать PaddleOCR enrichment")
    ap.add_argument("--no_aggressive_decode", action="store_true",
                    help="не запускать медленный aggressive barcode/QR decode")
    ap.add_argument("--sharpest_gated", action="store_true",
                    help="доп. проход sharpest-frame + motion gate (эксперимент)")
    ap.add_argument("--fuzzy_ocr_catalog", action="store_true",
                    help="R4: fuzzy catalog по OCR-трека (только 25_12-20, эксперимент)")
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG),
                    help="локальный каталог SKU для enrichment")
    ap.add_argument("--workdir", default=None,
                    help="папка для промежуточных файлов (по умолчанию temp)")
    args = ap.parse_args()

    # подсказка по DYLD для pyzbar (macOS + brew zbar)
    if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
        os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")

    pipe = PriceTagPipeline(weights=args.weights,
                            use_llm=not args.no_llm,
                            use_paddle=not args.no_paddle,
                            use_aggressive_decode=not args.no_aggressive_decode,
                            use_sharpest_gated=args.sharpest_gated,
                            use_fuzzy_ocr_catalog=args.fuzzy_ocr_catalog,
                            catalog_path=args.catalog,
                            workdir=args.workdir)
    df = pipe.process_video(args.video)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
