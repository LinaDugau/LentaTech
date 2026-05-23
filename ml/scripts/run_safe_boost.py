"""Decode + apply_qr + match + eval."""
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
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--ocr-src", type=Path, default=ROOT / "ml" / "output" / "_heavy_all3")
    ap.add_argument("--final-src", type=Path, default=ROOT / "ml" / "output")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = args.workdir or (ROOT / "ml" / "output" / f"_safe_boost_{ts}")
    work.mkdir(parents=True, exist_ok=True)
    cat = HYBRID if HYBRID.is_file() else CATALOG

    for v in VIDEOS:
        ocr_s = args.ocr_src / f"ocr_qr_{v}.csv"
        fin_s = args.final_src / f"final_eval_{v}.csv"
        if ocr_s.is_file():
            shutil.copy2(ocr_s, work / f"ocr_qr_{v}.csv")
        if fin_s.is_file():
            shutil.copy2(fin_s, work / f"final_eval_{v}.csv")

    for v in VIDEOS:
        ocr = work / f"ocr_qr_{v}.csv"
        gt = ROOT / "Данные" / v / f"{v}.csv"
        if not ocr.is_file():
            continue
        for script, extra in (
            ("sharpest_gated_decode.py", ["--files", str(ocr)]),
            ("whole_frame_codes2.py", ["--files", str(ocr)]),
            ("aggressive_decode.py", ["--files", str(ocr)]),
            ("temporal_video_barcode.py", ["--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)]),
            ("barcode_zone_decode.py", ["--files", str(ocr), "--catalog", str(CATALOG), "--crop-only", "--top-k", "8", "--fast"]),
            ("partial_ean_recovery.py", ["--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)]),
        ):
            _run([PY, str(SCRIPTS / script)] + extra, f"{script} {v}")

        if gt.is_file():
            _run([PY, str(SCRIPTS / "sharpest_crop_barcode.py"),
                  "--files", str(ocr), "--gt", str(gt), "--catalog", str(CATALOG)],
                 f"sharpest_crop {v}")

    _run([PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
          "--final-dir", str(work), "--ocr-dir", str(work)], "apply_qr")

    for v in VIDEOS:
        final = work / f"final_eval_{v}.csv"
        if final.is_file():
            _run([PY, str(SCRIPTS / "match_to_catalog.py"),
                  "--final", str(final), "--catalog", str(cat),
                  "--threshold", "0.70", "--overwrite"],
                 f"match {v}")

    _run([PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)], "postprocess")
    _run([PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)], "eval")
    print(f"\nDone → {work}")


if __name__ == "__main__":
    main()
