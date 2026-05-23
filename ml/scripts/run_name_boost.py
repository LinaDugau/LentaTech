"""Улучшение product_name."""
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
OUTPUT = ROOT / "ml" / "output"
SUBMISSION = ROOT / "ml" / "submission"
CAT = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
GGUF = (
    ROOT / "ml" / "weights" / "qwen2.5-3b-q4_k_m.gguf",
    ROOT / "weights" / "qwen2.5-3b-q4_k_m.gguf",
)


def _run(cmd: list, desc: str) -> None:
    print(f"\n>>> {desc}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if not args.dry_run:
        snap = OUTPUT / f"_backup_pre_name_boost_{ts}"
        snap.mkdir(parents=True, exist_ok=True)
        for v in ("25_12-20", "26_12-20", "43_15"):
            p = OUTPUT / f"final_eval_{v}.csv"
            if p.is_file():
                shutil.copy2(p, snap / p.name)
        print(f"Backup → {snap}")

    fuzzy_jobs = (
        ("26_12-20", "beverage", 0.84, False),
        ("43_15", "all", 0.86, True),
        ("25_12-20", "beverage", 0.88, False),
    )
    for video, subset, thr, no_brand in fuzzy_jobs:
        cmd = [
            PY, str(SCRIPTS / "fuzzy_ocr_catalog.py"),
            "--video-only", video,
            "--catalog", str(CAT),
            "--subset", subset,
            "--threshold", str(thr),
            "--name-only",
        ]
        if no_brand:
            cmd.append("--no-require-brand")
        if args.dry_run:
            print("would run:", " ".join(cmd))
            continue
        _run(cmd, f"fuzzy name-only {video} thr={thr}")

    gguf = next((p for p in GGUF if p.is_file()), None)
    if not args.skip_llm and not args.dry_run:
        for video in ("25_12-20", "26_12-20", "43_15"):
            final = OUTPUT / f"final_eval_{video}.csv"
            ocr = OUTPUT / f"ocr_qr_{video}.csv"
            cmd = [
                PY, str(SCRIPTS / "llm_product_name.py"),
                "--final", str(final), "--ocr", str(ocr),
                "--only-empty", "--min-confidence", "25",
            ]
            if gguf:
                cmd.extend(["--gguf", str(gguf)])
            try:
                _run(cmd, f"LLM {video}")
            except subprocess.CalledProcessError:
                print(f"  LLM skipped {video}")

    if not args.dry_run:
        cat_arg = ["--catalog", str(CAT)] if CAT.is_file() else []
        _run(
            [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(OUTPUT),
             "--finalize-names", *cat_arg],
            "finalize names",
        )
        SUBMISSION.mkdir(parents=True, exist_ok=True)
        for v in ("25_12-20", "26_12-20", "43_15"):
            src = OUTPUT / f"final_eval_{v}.csv"
            if src.is_file():
                shutil.copy2(src, SUBMISSION / src.name)
        _run([PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(OUTPUT)], "eval")


if __name__ == "__main__":
    main()
