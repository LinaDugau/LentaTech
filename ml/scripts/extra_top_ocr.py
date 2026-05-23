"""OCR верхней зоны ценника (product_name / бренд)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))
from video_resolve import find_video  # noqa: E402

_READER = None
_PADDLE = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    return _READER


def _limit_paddle_threads() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def get_paddle():
    global _PADDLE
    if _PADDLE is None:
        _limit_paddle_threads()
        from paddleocr import PaddleOCR
        _PADDLE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="ru",
            device="cpu",
        )
    return _PADDLE


class _VideoReader:
    """Lazy frame reader с кэшем последнего кадра."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cur_idx = -1
        self._cur_frame: np.ndarray | None = None

    def frame_at_ms(self, ts_ms: float) -> np.ndarray | None:
        idx = int(round(float(ts_ms) * self.fps / 1000.0))
        if idx != self._cur_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._cur_idx = idx
            self._cur_frame = frame
        return self._cur_frame

    def close(self) -> None:
        self.cap.release()


def _pad_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_pct: float = 0.06):
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_pct), int(bh * pad_pct)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def _crop_from_row(row, reader: _VideoReader | None, pad_pct: float = 0.06) -> np.ndarray | None:
    if reader is not None:
        frame = reader.frame_at_ms(float(row["frame_ts_ms"]))
        if frame is not None:
            x1, y1, x2, y2 = (
                int(row["x_min_orig"]),
                int(row["y_min_orig"]),
                int(row["x_max_orig"]),
                int(row["y_max_orig"]),
            )
            x1, y1, x2, y2 = _pad_bbox(x1, y1, x2, y2, reader.w, reader.h, pad_pct)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    cp = row.get("hires_crop") or row.get("crop_path")
    if isinstance(cp, str) and cp.strip():
        p = ROOT / cp
        if p.is_file():
            return cv2.imread(str(p))
    return None


def _unsharp(gray: np.ndarray, amount: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess_top(img: np.ndarray, top_pct: float = 0.65, upscale: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    top = img[: max(8, int(h * top_pct)), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = _unsharp(eq)
    if upscale > 1:
        eq = cv2.resize(
            eq,
            (eq.shape[1] * upscale, eq.shape[0] * upscale),
            interpolation=cv2.INTER_CUBIC,
        )
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def _rotations(img: np.ndarray, full: bool = False):
    yield img
    yield cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if full:
        yield cv2.rotate(img, cv2.ROTATE_180)
        yield cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _easyocr_text(img: np.ndarray, min_conf: float = 0.25, full_rot: bool = False) -> str:
    best_score = -1.0
    best_tokens: list[str] = []
    for rot in _rotations(img, full=full_rot):
        try:
            results = get_reader().readtext(rot, detail=1, paragraph=False)
        except Exception:
            continue
        tokens = [str(t).strip() for (_, t, c) in results if t and float(c) >= min_conf]
        score = sum(float(c) for (_, _, c) in results)
        if score > best_score and tokens:
            best_score = score
            best_tokens = tokens
    return "\n".join(best_tokens)


def _tile_ocr(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return ""
    parts: list[str] = []
    for y0, y1 in ((0, h // 2), (h // 2, h)):
        for x0, x1 in ((0, w // 2), (w // 2, w)):
            tile = img[y0:y1, x0:x1]
            if tile.shape[0] < 12 or tile.shape[1] < 12:
                continue
            t = _easyocr_text(tile, min_conf=0.2)
            if t.strip():
                parts.append(t.strip())
    return "\n".join(parts)


def _paddle_text(img: np.ndarray, min_score: float = 0.35) -> str:
    try:
        result = get_paddle().predict(img)
    except Exception:
        return ""
    if not result:
        return ""
    res = result[0].json.get("res", {}) if hasattr(result[0], "json") else {}
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    kept = [str(t) for t, s in zip(texts, scores) if s >= min_score and str(t).strip()]
    return "\n".join(kept)


def ocr_top_image(img: np.ndarray, use_paddle: bool = False, full_rot: bool = False) -> str:
    if img is None or img.size == 0:
        return ""
    proc = preprocess_top(img)
    if proc.shape[0] < 16 or proc.shape[1] < 16:
        return ""
    chunks: list[str] = []
    main = _easyocr_text(proc, full_rot=full_rot)
    if main.strip():
        chunks.append(main.strip())
    if len(main.strip()) < 40:
        tiled = _tile_ocr(proc)
        if tiled.strip() and tiled.strip() != main.strip():
            chunks.append(tiled.strip())
    if use_paddle:
        pt = _paddle_text(proc)
        if pt.strip():
            chunks.append(pt.strip())
    seen: set[str] = set()
    out: list[str] = []
    for block in chunks:
        for line in block.splitlines():
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(line.strip())
    return "\n".join(out)


def _blur_var(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_track_rows(df: pd.DataFrame, sharpest: bool = False, reader: _VideoReader | None = None) -> pd.DataFrame:
    """Один representative row на track — max area или max Laplacian (sharpest)."""
    if "track_id" not in df.columns:
        return df
    if not sharpest or reader is None:
        areas = (
            (df["x_max_orig"].astype(float) - df["x_min_orig"].astype(float))
            * (df["y_max_orig"].astype(float) - df["y_min_orig"].astype(float))
        )
        tmp = df.copy()
        tmp["_area"] = areas
        return tmp.sort_values("_area", ascending=False).drop_duplicates("track_id", keep="first")

    best: dict[int, tuple[float, int]] = {}
    for idx, row in df.iterrows():
        tid = int(row["track_id"])
        img = _crop_from_row(row, reader)
        if img is None or img.size == 0:
            score = -1.0
        else:
            score = _blur_var(img)
        prev = best.get(tid)
        if prev is None or score > prev[0]:
            best[tid] = (score, idx)
    if not best:
        return df.iloc[0:0]
    picks = [idx for _, idx in best.values()]
    return df.loc[picks].copy()


def _track_ids_from_spatio(
    rows_df: pd.DataFrame,
    ocr_df: pd.DataFrame,
    *,
    ts_col: str = "frame_timestamp",
    x_col: str = "x_min",
    y_col: str = "y_min",
) -> set[int]:
    tids: set[int] = set()
    for _, r in rows_df.iterrows():
        ts = float(r.get(ts_col, 0))
        x0 = float(str(r.get(x_col, 0)).replace(",", "."))
        y0 = float(str(r.get(y_col, 0)).replace(",", "."))
        cand = ocr_df[
            (ocr_df["frame_ts_ms"].astype(float).sub(ts).abs() < 15)
            & (ocr_df["x_min_orig"].astype(float).sub(x0).abs() < 12)
            & (ocr_df["y_min_orig"].astype(float).sub(y0).abs() < 12)
        ]
        if not cand.empty and "track_id" in cand.columns:
            tids.add(int(cand.iloc[0]["track_id"]))
    return tids


def _track_ids_from_final(final_path: Path, ocr_path: Path) -> set[int]:
    return _track_ids_from_spatio(pd.read_csv(final_path), pd.read_csv(ocr_path))


def _to_float(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _track_ids_from_gt(gt_path: Path, ocr_path: Path, min_iou: float = 0.25) -> set[int]:
    """GT bbox → track_id через max IoU с OCR (frame_ts в GT часто 0)."""
    gt = pd.read_csv(gt_path)
    ocr = pd.read_csv(ocr_path)
    if ocr.empty or gt.empty or "track_id" not in ocr.columns:
        return set()

    ocr_boxes = [
        (
            int(row.track_id),
            (
                float(row.x_min_orig), float(row.y_min_orig),
                float(row.x_max_orig), float(row.y_max_orig),
            ),
        )
        for row in ocr.itertuples(index=False)
    ]
    tids: set[int] = set()
    for _, g in gt.iterrows():
        gb = (
            _to_float(g.get("x_min")), _to_float(g.get("y_min")),
            _to_float(g.get("x_max")), _to_float(g.get("y_max")),
        )
        if gb[2] <= gb[0] or gb[3] <= gb[1]:
            continue
        best_iou = 0.0
        best_tid: int | None = None
        for tid, ob in ocr_boxes:
            v = _iou(gb, ob)
            if v > best_iou:
                best_iou = v
                best_tid = tid
        if best_tid is not None and best_iou >= min_iou:
            tids.add(best_tid)
    return tids


def process(
    csv_path: Path,
    use_paddle: bool = False,
    force: bool = False,
    per_track: bool = False,
    source: str = "auto",
    limit: int | None = None,
    full_rot: bool = False,
    track_ids: set[int] | None = None,
    sharpest: bool = False,
) -> None:
    full_df = pd.read_csv(csv_path)
    if full_df.empty:
        return
    if "top_extra_text" not in full_df.columns:
        full_df["top_extra_text"] = ""

    df = full_df
    if track_ids:
        df = full_df[full_df["track_id"].astype(int).isin(track_ids)].copy()
        if df.empty:
            print("  no rows for requested track_ids")
            return
        print(f"  filtered to {df.track_id.nunique()} tracks ({len(df)} rows)")

    video_name = str(df.iloc[0].get("video", "") or "").strip()
    reader: _VideoReader | None = None
    if source in ("auto", "video") and video_name:
        vpath = find_video(video_name)
        if vpath is not None:
            reader = _VideoReader(vpath)
            print(f"  source video: {vpath} ({reader.w}x{reader.h})")
        elif source == "video":
            print(f"  video not found: {video_name}")

    work = _pick_track_rows(df, sharpest=sharpest, reader=reader) if per_track else df
    if sharpest and per_track:
        print(f"  sharpest-frame per track: {len(work)} rows")
    if limit:
        work = work.head(limit)

    track_text: dict[int, str] = {}
    n_new = 0

    try:
        for _, row in tqdm(work.iterrows(), total=len(work), desc=csv_path.stem):
            idx = row.name
            existing = str(full_df.at[idx, "top_extra_text"] or "").strip()
            if existing and not force:
                continue

            use_reader = reader if source != "hires" else None
            img = _crop_from_row(row, use_reader)
            text = ocr_top_image(img, use_paddle=use_paddle, full_rot=full_rot)
            tid = int(row["track_id"]) if per_track and "track_id" in row.index else None
            if tid is not None:
                track_text[tid] = text
            else:
                full_df.at[idx, "top_extra_text"] = text
            n_new += 1

        if per_track and track_text:
            for idx, row in full_df.iterrows():
                tid = int(row["track_id"])
                if tid not in track_text:
                    continue
                if str(full_df.at[idx, "top_extra_text"] or "").strip() and not force:
                    continue
                full_df.at[idx, "top_extra_text"] = track_text[tid]
        elif not per_track:
            for idx in work.index:
                if idx in full_df.index and "top_extra_text" in work.columns:
                    full_df.at[idx, "top_extra_text"] = work.at[idx, "top_extra_text"]
    finally:
        if reader is not None:
            reader.close()

    def _merge(row):
        base = str(row.get("ocr_text", "") or "").strip()
        top = str(row.get("top_extra_text", "") or "").strip()
        if top and top not in base:
            return (base + "\n" + top).strip() if base else top
        return base

    full_df["ocr_text"] = full_df.apply(_merge, axis=1)
    full_df.to_csv(csv_path, index=False)
    total_nonempty = sum(1 for v in full_df["top_extra_text"] if str(v).strip())
    print(f"  {csv_path.name}: new={n_new}, top_nonempty={total_nonempty}/{len(full_df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--paddle", action="store_true", help="доп. Paddle на upscaled top")
    ap.add_argument("--force", action="store_true", help="пересчитать даже если top_extra_text есть")
    ap.add_argument("--per-track", action="store_true", help="один OCR на track_id (быстрее)")
    ap.add_argument("--source", choices=("auto", "video", "hires"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="только первые N строк work-set")
    ap.add_argument("--full-rot", action="store_true", help="4 поворота вместо 2")
    ap.add_argument("--final", type=str, default=None,
                    help="final_eval CSV → OCR только треков из final")
    ap.add_argument("--gt", type=str, default=None,
                    help="GT CSV → OCR только треков из разметки (57 для 25)")
    ap.add_argument("--sharpest", action="store_true",
                    help="с per-track: кадр с max Laplacian, не max area")
    args = ap.parse_args()
    files = [Path(p) for p in args.files] if args.files else sorted(
        (ROOT / "ml" / "output").glob("ocr_qr_*.csv")
    )
    files = [f for f in files if "_2-10" not in f.name and f.stem != "ocr_qr_input"]
    for f in files:
        print(f"\n=== {f.name} ===")
        tids = None
        if args.gt:
            tids = _track_ids_from_gt(Path(args.gt), f)
            print(f"  --gt: {len(tids)} track_ids")
        elif args.final:
            tids = _track_ids_from_final(Path(args.final), f)
            print(f"  --final: {len(tids)} track_ids")
        process(
            f,
            use_paddle=args.paddle,
            force=args.force,
            per_track=args.per_track,
            source=args.source,
            limit=args.limit,
            full_rot=args.full_rot,
            track_ids=tids,
            sharpest=args.sharpest,
        )


if __name__ == "__main__":
    main()
