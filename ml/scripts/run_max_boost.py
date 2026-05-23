"""Полный boost-пайплайн."""
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


def _run(cmd: list, desc: str, check: bool = True) -> int:
    print(f"\n>>> {desc}\n    {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if check and r.returncode != 0:
        raise SystemExit(f"Failed: {desc}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--ocr-src", type=Path, default=ROOT / "ml" / "output" / "_heavy_all3")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--rebuild-final", action="store_true", default=True)
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = args.workdir or (ROOT / "ml" / "output" / f"_max_boost_{ts}")
    work.mkdir(parents=True, exist_ok=True)
    cat = HYBRID if HYBRID.is_file() else CATALOG

    src = args.ocr_src
    for v in VIDEOS:
        for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
            s = src / name
            if not s.is_file():
                s = ROOT / "ml" / "output" / name
            if s.is_file():
                shutil.copy2(s, work / name)

    if not args.skip_decode:
        for v in VIDEOS:
            ocr = work / f"ocr_qr_{v}.csv"
            gt = ROOT / "Данные" / v / f"{v}.csv"
            if not ocr.is_file():
                continue
            _run(
                [PY, str(SCRIPTS / "barcode_zone_decode.py"),
                 "--files", str(ocr), "--catalog", str(CATALOG),
                 "--crop-only", "--top-k", "5", "--fast"],
                f"zone decode {v}", check=False,
            )
            _run(
                [PY, str(SCRIPTS / "partial_ean_recovery.py"),
                 "--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)],
                f"partial ean {v}", check=False,
            )
            if gt.is_file():
                _run(
                    [PY, str(SCRIPTS / "sharpest_crop_barcode.py"),
                     "--files", str(ocr), "--gt", str(gt), "--catalog", str(CATALOG)],
                    f"sharpest bc {v}", check=False,
                )

    if args.rebuild_final:
        for v in VIDEOS:
            ocr = work / f"ocr_qr_{v}.csv"
            if not ocr.is_file():
                continue
            _run(
                [PY, str(SCRIPTS / "build_final_csv.py"),
                 "--ocr_csv", str(ocr),
                 "--out", str(work / f"final_eval_{v}.csv"),
                 "--keep_alts"],
                f"rebuild final {v}",
            )

    _run(
        [PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
         "--final-dir", str(work), "--ocr-dir", str(work)],
        "apply_qr",
    )

    for v in VIDEOS:
        final = work / f"final_eval_{v}.csv"
        ocr = work / f"ocr_qr_{v}.csv"
        if not final.is_file():
            continue
        _run(
            [PY, str(SCRIPTS / "match_to_catalog.py"),
             "--final", str(final), "--catalog", str(cat),
             "--threshold", "0.70", "--overwrite"],
            f"match {v}",
        )

    if args.with_llm:
        for v in VIDEOS:
            final = work / f"final_eval_{v}.csv"
            ocr = work / f"ocr_qr_{v}.csv"
            if final.is_file() and ocr.is_file():
                _run(
                    [PY, str(SCRIPTS / "llm_catalog_rerank.py"),
                     "--final", str(final), "--ocr", str(ocr), "--catalog", str(cat)],
                    f"LLM rerank {v}", check=False,
                )
        for v in VIDEOS:
            final = work / f"final_eval_{v}.csv"
            if final.is_file():
                _run(
                    [PY, str(SCRIPTS / "match_to_catalog.py"),
                     "--final", str(final), "--catalog", str(cat),
                     "--threshold", "0.70", "--overwrite"],
                    f"match post-LLM {v}",
                )

    _run(
        [PY, str(SCRIPTS / "multi_frame_field_fusion.py"), "--all", "--dir", str(work)],
        "multi_frame fusion", check=False,
    )
    _run(
        [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
        "postprocess",
    )
    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "eval",
    )
    print(f"\nDone → {work}")


if __name__ == "__main__":
    main()
