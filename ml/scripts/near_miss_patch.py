"""Near-miss boost для 26_12-20."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _copy_src(work: Path, src: Path, videos: list[str]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for v in videos:
        for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
            p = src / name
            if p.is_file():
                shutil.copy2(p, work / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=ROOT / "ml" / "output" / "_day21_nearmiss26")
    ap.add_argument("--from-dir", type=Path, default=ROOT / "ml" / "output" / "_heavy_all3")
    ap.add_argument("--videos", nargs="*", default=["26_12-20"])
    ap.add_argument("--fuzzy-threshold", type=float, default=0.90)
    ap.add_argument("--skip-fuzzy", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir)
    src = Path(args.from_dir)
    _copy_src(work, src, args.videos)
    print(f"Workdir: {work} (from {src})")

    for v in args.videos:
        final = work / f"final_eval_{v}.csv"
        ocr = work / f"ocr_qr_{v}.csv"
        if final.is_file() and ocr.is_file():
            _run(
                [PY, str(SCRIPTS / "patch_final_from_ocr.py"), "--video", v, "--dir", str(work)],
                f"patch_final {v}",
            )

    _run(
        [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
        "final_postprocess",
    )

    if not args.skip_fuzzy:
        for v in args.videos:
            final = work / f"final_eval_{v}.csv"
            ocr = work / f"ocr_qr_{v}.csv"
            if not final.is_file() or not ocr.is_file():
                continue
            _run(
                [
                    PY, str(SCRIPTS / "fuzzy_ocr_catalog.py"),
                    "--final", str(final),
                    "--ocr", str(ocr),
                    "--threshold", str(args.fuzzy_threshold),
                    "--video-only", v,
                    "--subset", "all",
                ],
                f"fuzzy catalog (barcode fill) {v} thr={args.fuzzy_threshold}",
            )
            _run(
                [
                    PY, str(SCRIPTS / "match_to_catalog.py"),
                    "--final", str(final),
                    "--catalog", str(ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"),
                    "--threshold", "0.70",
                    "--overwrite",
                ],
                f"match_to_catalog {v}",
            )
            _run(
                [PY, str(SCRIPTS / "final_postprocess.py"), "--final", str(final)],
                f"postprocess {v}",
            )

    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "evaluate_official",
    )
    print(f"\nDone. Workdir: {work}")


if __name__ == "__main__":
    main()
