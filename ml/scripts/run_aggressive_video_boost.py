"""Aggressive decode на одном видео."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable
CATALOG = ROOT / "db_hack.csv"
HYBRID = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"


def _run(cmd: list, desc: str) -> None:
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--video-decode", action="store_true")
    ap.add_argument("--undistort", action="store_true")
    args = ap.parse_args()
    v = args.video
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    cat = HYBRID if HYBRID.is_file() else CATALOG

    for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
        src = ROOT / "ml" / "output" / "_heavy_all3" / name
        if not src.is_file():
            src = ROOT / "ml" / "output" / name
        shutil.copy2(src, work / name)

    ocr = work / f"ocr_qr_{v}.csv"
    gt = ROOT / "Данные" / v / f"{v}.csv"

    _run([PY, str(SCRIPTS / "sharpest_gated_decode.py"), "--files", str(ocr)],
         "sharpest_gated_decode")
    _run([PY, str(SCRIPTS / "whole_frame_codes2.py"), "--files", str(ocr)],
         "whole_frame_codes2")
    _run([PY, str(SCRIPTS / "aggressive_decode.py"), "--files", str(ocr)],
         "aggressive_decode")
    _run([PY, str(SCRIPTS / "temporal_video_barcode.py"),
          "--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)],
         "temporal_video_barcode")

    zone = [PY, str(SCRIPTS / "barcode_zone_decode.py"),
            "--files", str(ocr), "--catalog", str(CATALOG), "--top-k", "8"]
    if args.video_decode:
        zone.append("--fast")
    else:
        zone.extend(["--crop-only", "--fast"])
    if args.undistort:
        zone.append("--undistort")
    _run(zone, "barcode_zone_decode")

    if gt.is_file():
        _run([PY, str(SCRIPTS / "sharpest_crop_barcode.py"),
              "--files", str(ocr), "--gt", str(gt), "--catalog", str(CATALOG),
              "--all-frames", "--fast"],
             "sharpest_crop_barcode all-frames")

    _run([PY, str(SCRIPTS / "partial_ean_recovery.py"),
          "--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)],
         "partial_ean")

    _run([PY, str(SCRIPTS / "build_final_csv.py"),
          "--ocr_csv", str(ocr),
          "--out", str(work / f"final_eval_{v}.csv"), "--keep_alts"],
         "rebuild final")

    _run([PY, str(SCRIPTS / "apply_qr_to_final.py"),
          "--final-dir", str(work), "--ocr-dir", str(work)],
         "apply_qr")

    _run([PY, str(SCRIPTS / "match_to_catalog.py"),
          "--final", str(work / f"final_eval_{v}.csv"),
          "--catalog", str(cat), "--threshold", "0.70", "--overwrite"],
         "match_to_catalog")

    _run([PY, str(SCRIPTS / "final_postprocess.py"),
          "--final", str(work / f"final_eval_{v}.csv"), "--dir", str(work)],
         "postprocess")

    _run([PY, str(SCRIPTS / "evaluate_official.py"),
          "--pred-dir", str(work), "--videos", v],
         "eval")
    print(f"\nDone → {work}")


if __name__ == "__main__":
    main()
