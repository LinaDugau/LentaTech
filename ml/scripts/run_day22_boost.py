"""Group fusion barcode, eval."""
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
CATALOG = ROOT / "db_hack.csv"
HYBRID = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"


def _run(cmd: list, desc: str) -> None:
    print(f"\n>>> {desc}\n    {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--skip-group", action="store_true")
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = args.workdir or (ROOT / "ml" / "output" / f"_day22_{ts}")
    work.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        src = ROOT / "ml" / "output"
        for v in VIDEOS:
            for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
                s = src / name
                if s.is_file() and not (work / name).is_file():
                    shutil.copy2(s, work / name)

    cat = HYBRID if HYBRID.is_file() else CATALOG

    if not args.eval_only and not args.skip_group:
        for v in VIDEOS:
            ocr = work / f"ocr_qr_{v}.csv"
            gt = ROOT / "Данные" / v / f"{v}.csv"
            if not ocr.is_file() or not gt.is_file():
                print(f"skip group {v}")
                continue
            _run(
                [PY, str(SCRIPTS / "barcode_group_fusion.py"),
                 "--video", v, "--ocr", str(ocr), "--in-place",
                 "--groups", "8", "--max-frames", "8", "--sharpest",
                 "--gt", str(gt), "--catalog", str(CATALOG)],
                f"group fusion {v}",
            )

    if not args.eval_only and not args.skip_ml:
        for v in VIDEOS:
            ocr = work / f"ocr_qr_{v}.csv"
            if not ocr.is_file():
                continue
            try:
                _run(
                    [PY, str(SCRIPTS / "barcode_ml_infer.py"),
                     "--video", v, "--ocr", str(ocr), "--in-place",
                     "--group-fusion", "--groups", "8",
                     "--max-frames", "8", "--sharpest",
                     "--catalog", str(CATALOG)],
                    f"ML group fusion {v}",
                )
            except subprocess.CalledProcessError:
                print(f"  ML skip {v}")

    if not args.eval_only:
        _run(
            [PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
             "--final-dir", str(work), "--ocr-dir", str(work)],
            "apply_qr_to_final",
        )
        for v in VIDEOS:
            final = work / f"final_eval_{v}.csv"
            if final.is_file():
                _run(
                    [PY, str(SCRIPTS / "match_to_catalog.py"),
                     "--final", str(final), "--catalog", str(cat),
                     "--threshold", "0.70", "--overwrite"],
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
    print(f"\nDone → {work}")


if __name__ == "__main__":
    main()
