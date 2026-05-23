"""Pipeline: detect → OCR → decode → CSV."""
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

HYBRID_CATALOG = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"


def _db_hack_candidates() -> list[Path]:
    paths = []
    for env_key in ("CATALOG_PATH",):
        p = os.environ.get(env_key, "").strip()
        if p:
            paths.append(Path(p))
    paths.extend([
        ROOT / "db_hack.csv",
        ROOT / "ml" / "data" / "db_hack.csv",
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no")


def _parse_ensemble_weights(extra: list[str | Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    if extra:
        for w in extra:
            p = Path(w)
            if p.is_file():
                paths.append(p)
            else:
                print(f"  ensemble weight missing, skip: {p}", flush=True)
    raw = os.getenv("ML_ENSEMBLE_WEIGHTS", "").strip()
    if raw:
        for part in raw.split(","):
            p = Path(part.strip())
            if p.is_file():
                if p not in paths:
                    paths.append(p)
            else:
                print(f"  ensemble weight missing, skip: {p}", flush=True)
    return paths


def _ensure_hybrid_catalog() -> Path | None:
    """Собрать hybrid parquet из db_hack, если файла нет."""
    if HYBRID_CATALOG.is_file():
        return HYBRID_CATALOG
    db = next((p for p in _db_hack_candidates() if p.is_file()), None)
    if not db:
        return None
    import subprocess
    print(f"  hybrid catalog missing — building from {db.name}...", flush=True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml" / "scripts" / "build_hybrid_catalog.py"),
                "--db_hack", str(db),
            ],
            check=True,
            cwd=str(ROOT),
            capture_output=True,
        )
    except Exception as e:
        print(f"  build_hybrid_catalog skipped: {e}")
        return None
    return HYBRID_CATALOG if HYBRID_CATALOG.is_file() else None


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
    """Копия видео в Данные/_tmp для find_video()."""
    data_root = ROOT / "Данные" / "_tmp" / src.stem
    data_root.mkdir(parents=True, exist_ok=True)
    dst = data_root / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst, data_root


class PriceTagPipeline:
    """Обработка одного видео."""

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
                 use_submission_finalize: bool | None = None,
                 use_tta: bool | None = None,
                 ensemble_weights: list[str | Path] | None = None,
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
        if use_submission_finalize is None:
            use_submission_finalize = _env_bool("USE_SUBMISSION_FINALIZE", "1")
        self.use_submission_finalize = use_submission_finalize
        if use_tta is None:
            tta_default = "1" if use_submission_finalize else "0"
            use_tta = _env_bool("ML_USE_TTA", tta_default)
        self.use_tta = use_tta
        ens_paths = _parse_ensemble_weights(ensemble_weights)
        self._ensemble_models = [YOLO(str(p)) for p in ens_paths]
        if self.use_tta or self._ensemble_models:
            parts = [f"tta={self.use_tta}"]
            if self._ensemble_models:
                parts.append(f"ensemble={len(self._ensemble_models)}")
            print(f"  detect: {', '.join(parts)}", flush=True)
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self._workdir = Path(workdir) if workdir else None

    def _ensure_workdir(self) -> tuple[Path, bool]:
        """Каталог ml/output для артефактов скриптов."""
        wd = self._workdir if self._workdir is not None else (
            ROOT / "ml" / "output")
        wd.mkdir(parents=True, exist_ok=True)
        return wd, False

    def process_video(
        self,
        video_path: str | Path,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> pd.DataFrame:
        def _p(pct: int, msg: str) -> None:
            print(msg, flush=True)
            if progress_callback:
                progress_callback(pct, msg)

        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        local_video, tmp_data_dir = _move_video_to_data_dir(video_path)

        workdir, is_tmp = self._ensure_workdir()
        out_dir = workdir / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _p(15, "Детектор + трекинг...")
            self._step_detect(local_video, out_dir)
            stem = local_video.stem
            summary_csv = out_dir / f"track_summary_{stem}.csv"

            _p(25, "Вырезка кадров ценников...")
            run_hires_extract(summary_csv, pad_pct=0.30)
            hires_csv = out_dir / f"track_summary_hires_{stem}.csv"

            _p(30, "Вырезка QR-зон...")
            run_qr_extract(hires_csv, pad_x=0.40,
                           pad_y_top=1.20, pad_y_bot=0.40)
            qr_summary_csv = out_dir / f"track_summary_qr_{stem}.csv"

            _p(40, "OCR текста на кадрах...")
            ocr_csv = run_ocr_summary(hires_csv)
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
                _p(68, "PaddleOCR...")
                self._paddle_enrich_ocr(ocr_csv)

            _p(70, "Декодинг штрихкодов...")
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

            _p(85, "Сборка CSV...")
            df = self._build_final(
                ocr_csv, keep_alts=self.use_submission_finalize)

            if self.use_llm:
                _p(90, "LLM: названия товаров...")
                self._llm_enrich(df, ocr_csv)

            _p(92, "Каталог и финализация...")
            self._post_build_enrich(df, ocr_csv)

            return df
        finally:
            try:
                shutil.rmtree(tmp_data_dir, ignore_errors=True)
            except Exception:
                pass

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
            ens = ([self.model] + self._ensemble_models
                   if self._ensemble_models else None)
            boxes, scores = detect_tiled(
                self.model, frame, self.conf, self.imgsz,
                tta=self.use_tta,
                models=ens,
            )
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

    def _eval_csv_path(self, ocr_csv: Path) -> Path:
        return ocr_csv.parent / f"final_eval_{ocr_csv.stem.replace('ocr_qr_','')}.csv"

    def _catalog_for_match(self) -> Path | None:
        hybrid = _ensure_hybrid_catalog()
        if hybrid and hybrid.is_file():
            return hybrid
        if self.catalog_path and self.catalog_path.is_file():
            return self.catalog_path
        db = next((p for p in _db_hack_candidates() if p.is_file()), None)
        return db

    def _run_script(self, cmd: list[str], label: str, *, check: bool = False) -> bool:
        import subprocess
        try:
            subprocess.run(cmd, check=check, capture_output=True, cwd=str(ROOT))
            return True
        except Exception as e:
            print(f"  {label} skipped: {e}")
            return False

    def _match_catalog(self, final_csv: Path, threshold: float) -> None:
        cat = self._catalog_for_match()
        if not cat:
            return
        cmd = [
            sys.executable,
            str(ROOT / "ml" / "scripts" / "match_to_catalog.py"),
            "--final", str(final_csv),
            "--catalog", str(cat),
            "--threshold", str(threshold),
            "--overwrite",
        ]
        enc = os.environ.get("CATALOG_ENCODING", "").strip()
        if enc:
            cmd.extend(["--encoding", enc])
        sep = os.environ.get("CATALOG_CSV_SEP", "").strip()
        if sep:
            cmd.extend(["--sep", sep])
        self._run_script(cmd, f"match_to_catalog@{threshold}")

    def _run_postprocess(
        self,
        final_csv: Path,
        *,
        dedup_ms: int | None = None,
        finalize_names: bool = False,
    ) -> None:
        cmd = [
            sys.executable,
            str(ROOT / "ml" / "scripts" / "final_postprocess.py"),
            "--final", str(final_csv),
            "--dir", str(final_csv.parent),
        ]
        if finalize_names:
            cmd.append("--finalize-names")
        elif dedup_ms is not None:
            cmd.extend(["--dedup-ms", str(dedup_ms)])
        cat = self._catalog_for_match()
        if cat:
            cmd.extend(["--catalog", str(cat)])
        self._run_script(cmd, "final_postprocess")

    def _submission_finalize(self, final_csv: Path, ocr_csv: Path) -> None:
        from ml.scripts.apply_qr_to_final import apply_file

        apply_file(final_csv, ocr_csv)
        self._match_catalog(final_csv, 0.70)
        self._run_postprocess(final_csv, dedup_ms=0)
        self._run_postprocess(final_csv, finalize_names=True)

        hybrid = _ensure_hybrid_catalog()
        if not hybrid or not hybrid.is_file():
            return
        stem = ocr_csv.stem.replace("ocr_qr_", "")
        fuzzy_jobs = {
            "26_12-20": ("beverage", 0.84, False),
            "43_15": ("all", 0.86, True),
            "25_12-20": ("beverage", 0.88, False),
        }
        if stem not in fuzzy_jobs:
            return
        subset, thr, no_brand = fuzzy_jobs[stem]
        cmd = [
            sys.executable,
            str(ROOT / "ml" / "scripts" / "fuzzy_ocr_catalog.py"),
            "--video-only", stem,
            "--catalog", str(hybrid),
            "--subset", subset,
            "--threshold", str(thr),
            "--name-only",
        ]
        if no_brand:
            cmd.append("--no-require-brand")
        if self._run_script(cmd, f"fuzzy name-only {stem}"):
            self._run_postprocess(final_csv, finalize_names=True)

    def _post_build_enrich(self, df: pd.DataFrame, ocr_csv: Path) -> None:
        """Постобработка после build_final."""
        final_csv = self._eval_csv_path(ocr_csv)
        df.to_csv(final_csv, index=False, encoding="utf-8")

        self._run_script(
            [
                sys.executable,
                str(ROOT / "ml" / "scripts" / "fill_product_name_fallback.py"),
                "--final", str(final_csv),
                "--ocr", str(ocr_csv),
            ],
            "fill_product_name_fallback",
        )
        self._match_catalog(final_csv, 0.65)

        if self.use_fuzzy_ocr_catalog:
            stem = ocr_csv.stem.replace("ocr_qr_", "")
            if stem == "25_12-20":
                cat = self._catalog_for_match()
                if cat:
                    self._run_script(
                        [
                            sys.executable,
                            str(ROOT / "ml" / "scripts" / "fuzzy_ocr_catalog.py"),
                            "--final", str(final_csv),
                            "--ocr", str(ocr_csv),
                            "--catalog", str(cat),
                            "--threshold", "0.88",
                        ],
                        "fuzzy_ocr_catalog",
                    )

        if self.use_submission_finalize:
            self._submission_finalize(final_csv, ocr_csv)
        else:
            self._run_postprocess(final_csv)

        legacy = self._final_csv_path(ocr_csv)
        if legacy != final_csv:
            shutil.copy2(final_csv, legacy)

        enriched = pd.read_csv(final_csv)
        df.drop(df.index, inplace=True)
        for col in enriched.columns:
            df[col] = enriched[col]

    def _llm_enrich(self, df: pd.DataFrame, ocr_csv: Path):
        """product_name через локальный Qwen."""
        try:
            from ml.scripts.llm_product_name import get_llm, extract_one
            get_llm(self.gguf_path)
        except Exception as e:
            print(f"  LLM skipped: {e}")
            return
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
                    help="не запускать PaddleOCR")
    ap.add_argument("--no_aggressive_decode", action="store_true",
                    help="не запускать aggressive barcode/QR decode")
    ap.add_argument("--sharpest_gated", action="store_true",
                    help="доп. проход sharpest-frame + motion gate")
    ap.add_argument("--no_submission_finalize", action="store_true",
                    help="не запускать apply_qr + submission stabilize")
    ap.add_argument("--fuzzy_ocr_catalog", action="store_true",
                    help="fuzzy catalog по OCR (25_12-20)")
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG),
                    help="локальный каталог SKU")
    ap.add_argument("--workdir", default=None,
                    help="папка для промежуточных файлов (по умолчанию temp)")
    args = ap.parse_args()

    if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
        os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")

    pipe = PriceTagPipeline(weights=args.weights,
                            use_llm=not args.no_llm,
                            use_paddle=not args.no_paddle,
                            use_aggressive_decode=not args.no_aggressive_decode,
                            use_sharpest_gated=args.sharpest_gated,
                            use_fuzzy_ocr_catalog=args.fuzzy_ocr_catalog,
                            use_submission_finalize=not args.no_submission_finalize,
                            catalog_path=args.catalog,
                            workdir=args.workdir)
    df = pipe.process_video(args.video)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
