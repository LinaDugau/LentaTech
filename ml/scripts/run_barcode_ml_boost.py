"""ML barcode: dataset → train → infer."""
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


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    print("    ", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-pretrain-gen", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--catalog-limit", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pretrain-epochs", type=int, default=6)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = args.workdir or (ROOT / "ml" / "output" / f"_ml_barcode_{ts}")
    work.mkdir(parents=True, exist_ok=True)

    data_merged = ROOT / "ml" / "data" / "barcode_strips_merged"
    pretrain_dir = ROOT / "ml" / "data" / "barcode_strips_pretrain"
    external_dir = ROOT / "ml" / "data" / "barcode_strips_external"

    if not args.skip_download:
        _run(
            [PY, str(SCRIPTS / "download_external_barcode_data.py"), "--only", "sbd_lr"],
            "download SBD LR",
        )

    if not args.skip_pretrain_gen:
        _run(
            [PY, str(SCRIPTS / "generate_catalog_pretrain_strips.py"),
             "--limit", str(args.catalog_limit), "--aug", "2"],
            "generate catalog pretrain strips",
        )

    _run(
        [PY, str(SCRIPTS / "import_external_barcode_strips.py"), "--sbd-limit", "2000"],
        "import external strips (SBD if downloaded)",
    )

    heavy = ROOT / "ml" / "output" / "_heavy_all3"
    ocr_src = heavy if heavy.is_dir() else ROOT / "ml" / "output"
    build_cmd = [
        PY, str(SCRIPTS / "build_barcode_strip_dataset.py"),
        "--include-pred-crops", "--ocr-dir", str(ocr_src),
        "--merge-pretrain", str(pretrain_dir),
        "--merge-external", str(external_dir),
        "--out", str(data_merged),
    ]
    _run(build_cmd, "build merged dataset")

    if not args.skip_train:
        _run(
            [PY, str(SCRIPTS / "train_barcode_strip.py"),
             "--data", str(data_merged),
             "--epochs", str(args.epochs),
             "--pretrain-epochs", str(args.pretrain_epochs),
             "--holdout-video", "26_12-20"],
            "train Slot-CNN",
        )

    for v in ("25_12-20", "26_12-20", "43_15"):
        src_ocr = ocr_src / f"ocr_qr_{v}.csv"
        if not src_ocr.is_file():
            src_ocr = ROOT / "ml" / "output" / f"ocr_qr_{v}.csv"
        if src_ocr.is_file():
            shutil.copy2(src_ocr, work / f"ocr_qr_{v}.csv")
        src_final = ROOT / "ml" / "output" / f"final_eval_{v}.csv"
        if src_final.is_file():
            shutil.copy2(src_final, work / f"final_eval_{v}.csv")

    catalog = ROOT / "db_hack.csv"
    for v in VIDEOS:
        ocr = work / f"ocr_qr_{v}.csv"
        if not ocr.is_file():
            continue
        _run(
            [PY, str(SCRIPTS / "barcode_ml_infer.py"),
             "--video", v, "--ocr", str(ocr), "--out", str(ocr),
             "--sharpest", "--max-frames", "10", "--min-votes", "2",
             "--catalog", str(catalog)],
            f"ML infer {v}",
        )

    _run(
        [PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
         "--final-dir", str(work), "--ocr-dir", str(work)],
        "apply_qr_to_final",
    )
    hybrid = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
    cat = hybrid if hybrid.is_file() else catalog
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
    print(f"\nDone. Workdir: {work}")


if __name__ == "__main__":
    main()
