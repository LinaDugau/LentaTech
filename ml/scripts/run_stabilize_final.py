"""Стабилизация final CSV."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable
VIDEOS = ("25_12-20", "26_12-20", "43_15")
HYBRID = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
CATALOG = ROOT / "db_hack.csv"


def _run(cmd: list, desc: str) -> None:
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--src", type=Path, default=ROOT / "ml" / "output")
    ap.add_argument("--min-confidence", type=int, default=15)
    ap.add_argument("--dedup-ms", type=int, default=2000)
    ap.add_argument("--rematch", action="store_true",
                    help="повторно match_to_catalog после стабилизации")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = args.workdir or (ROOT / "ml" / "output" / f"_stabilized_{ts}")
    work.mkdir(parents=True, exist_ok=True)
    cat = HYBRID if HYBRID.is_file() else CATALOG

    for v in VIDEOS:
        src = args.src / f"final_eval_{v}.csv"
        if src.is_file():
            shutil.copy2(src, work / f"final_eval_{v}.csv")
        ocr = args.src / f"ocr_qr_{v}.csv"
        if ocr.is_file():
            shutil.copy2(ocr, work / f"ocr_qr_{v}.csv")

    _run(
        [
            PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work),
            "--min-confidence", str(args.min_confidence),
            "--dedup-ms", str(args.dedup_ms),
        ],
        "stabilize + postprocess",
    )

    if args.rematch and cat.is_file():
        for v in VIDEOS:
            final = work / f"final_eval_{v}.csv"
            if final.is_file():
                _run(
                    [
                        PY, str(SCRIPTS / "match_to_catalog.py"),
                        "--final", str(final), "--catalog", str(cat),
                        "--threshold", "0.70", "--overwrite",
                    ],
                    f"rematch {v}",
                )

    _run([PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)], "eval")
    print(f"\nDone → {work}")


if __name__ == "__main__":
    main()
