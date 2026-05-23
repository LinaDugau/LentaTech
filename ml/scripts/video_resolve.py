"""Поиск исходного видео по имени файла под `Данные/`."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [ROOT / "Данные", ROOT / "Данные" / "Unlabeled"]


def _score_path(path: Path, name: str) -> tuple[int, str]:
    """Выше = лучше. Предпочитаем `Данные/<stem>/<name>`."""
    stem = Path(name).stem
    parent = path.parent.name
    if parent == stem:
        return (4, path.as_posix())
    if parent.lower() == "unlabeled":
        return (0, path.as_posix())
    if parent == "_tmp":
        return (1, path.as_posix())
    return (2, path.as_posix())


def find_video(name: str) -> Path | None:
    best: Path | None = None
    best_score = (-1, "")

    for d in DATA_DIRS:
        if not d.is_dir():
            continue
        direct = d / name
        if direct.is_file():
            score = _score_path(direct, name)
            if score > best_score:
                best_score = score
                best = direct
        for sub in sorted(d.glob("*")):
            if not sub.is_dir():
                continue
            cand = sub / name
            if not cand.is_file():
                continue
            score = _score_path(cand, name)
            if score > best_score:
                best_score = score
                best = cand
        tmp_root = d / "_tmp"
        if tmp_root.is_dir():
            for job_dir in sorted(tmp_root.iterdir()):
                if not job_dir.is_dir():
                    continue
                cand = job_dir / name
                if not cand.is_file():
                    continue
                score = _score_path(cand, name)
                if score > best_score:
                    best_score = score
                    best = cand
    return best
