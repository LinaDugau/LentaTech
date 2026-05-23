"""Spatio catalog resolve + eval."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable
VIDEOS = ("25_12-20", "26_12-20", "43_15")


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _copy(work: Path, src: Path, videos: tuple[str, ...]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for v in videos:
        for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
            p = src / name
            if p.is_file():
                shutil.copy2(p, work / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=ROOT / "ml" / "output" / "_spatio_catalog_boost")
    ap.add_argument("--from-dir", type=Path, default=ROOT / "ml" / "output")
    ap.add_argument("--ocr-dir", type=Path, default=ROOT / "ml" / "output" / "_heavy_all3",
                    help="OCR с bottom/top (heavy); final берётся из --from-dir")
    ap.add_argument("--videos", nargs="*", default=list(VIDEOS))
    ap.add_argument("--subset", default="all", choices=("beverage", "all"))
    ap.add_argument("--auto-threshold", type=float, default=0.88)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir)
    if not args.skip_copy:
        _copy(work, Path(args.from_dir), tuple(args.videos))
        ocr_src = Path(args.ocr_dir)
        for v in args.videos:
            ocr = ocr_src / f"ocr_qr_{v}.csv"
            if ocr.is_file():
                shutil.copy2(ocr, work / f"ocr_qr_{v}.csv")
    print(f"Workdir: {work}")

    catalog = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
    if not catalog.is_file():
        catalog = ROOT / "db_hack.csv"

    for v in args.videos:
        final = work / f"final_eval_{v}.csv"
        ocr = work / f"ocr_qr_{v}.csv"
        if not final.is_file() or not ocr.is_file():
            continue
        cmd = [
            PY, str(SCRIPTS / "spatio_catalog_resolve.py"),
            "--final", str(final),
            "--ocr", str(ocr),
            "--catalog", str(catalog),
            "--subset", args.subset,
            "--auto-threshold", str(args.auto_threshold),
        ]
        if not args.use_llm:
            cmd.append("--no-llm")
        _run(cmd, f"spatio resolve {v}")

        _run(
            [
                PY, str(SCRIPTS / "match_to_catalog.py"),
                "--final", str(final),
                "--catalog", str(catalog),
                "--threshold", "0.70",
                "--overwrite",
            ],
            f"match_to_catalog {v}",
        )

    _run(
        [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
        "final_postprocess",
    )
    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "evaluate_official",
    )
    print(f"\nDone. {work}")


if __name__ == "__main__":
    main()
