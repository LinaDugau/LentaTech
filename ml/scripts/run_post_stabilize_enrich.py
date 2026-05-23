"""Фильтр + catalog + LLM."""
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
OUTPUT = ROOT / "ml" / "output"
SUBMISSION = ROOT / "ml" / "submission"
GGUF_CANDIDATES = (
    ROOT / "ml" / "weights" / "qwen2.5-3b-q4_k_m.gguf",
    ROOT / "weights" / "qwen2.5-3b-q4_k_m.gguf",
)


def _resolve_gguf() -> Path | None:
    return next((p for p in GGUF_CANDIDATES if p.is_file()), None)


def _run(cmd: list, desc: str, *, check: bool = True) -> bool:
    print(f"\n>>> {desc}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-backup", type=Path,
                    default=OUTPUT / "_backup_pre_stabilize_20260522_1955")
    ap.add_argument("--dedup-ms", type=int, default=0,
                    help="0 = сохранить TARGET, 2000 = меньше строк")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--llm-only", action="store_true",
                    help="только LLM + finalize на текущих final_eval (без restore)")
    ap.add_argument("--llm-videos", nargs="+", default=list(VIDEOS))
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cat = HYBRID if HYBRID.is_file() else CATALOG
    gguf = args.gguf or _resolve_gguf()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    snap = OUTPUT / f"_backup_pre_enrich_{ts}"

    if not args.dry_run:
        snap.mkdir(parents=True, exist_ok=True)
        for v in VIDEOS:
            for d, name in ((OUTPUT, "output"), (SUBMISSION, "submission")):
                p = d / f"final_eval_{v}.csv"
                if p.is_file():
                    shutil.copy2(p, snap / f"{name}_final_eval_{v}.csv")
        if args.llm_only:
            print(f"Backup before LLM: {snap}")

    if not args.llm_only and args.src_backup.is_dir():
        print(f"Restore from {args.src_backup}")
        for v in VIDEOS:
            src = args.src_backup / f"final_eval_{v}.csv"
            if src.is_file():
                if not args.dry_run:
                    shutil.copy2(src, OUTPUT / f"final_eval_{v}.csv")
            else:
                print(f"  missing {src}")

    cat_arg = ["--catalog", str(cat)] if cat.is_file() else []
    if not args.llm_only:
        _run(
            [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(OUTPUT),
             "--dedup-ms", str(args.dedup_ms), *cat_arg],
            f"stabilize dedup={args.dedup_ms}",
            check=not args.dry_run,
        )

        for v in VIDEOS:
            final = OUTPUT / f"final_eval_{v}.csv"
            if not final.is_file():
                continue
            _run(
                [PY, str(SCRIPTS / "match_to_catalog.py"),
                 "--final", str(final), "--catalog", str(cat),
                 "--threshold", "0.70", "--overwrite"],
                f"match_to_catalog {v}",
                check=not args.dry_run,
            )

    if not args.skip_llm:
        if not gguf or not gguf.is_file():
            print(f"GGUF not found (tried: {GGUF_CANDIDATES})")
        else:
            print(f"LLM weights: {gguf}")
        for v in args.llm_videos:
            final = OUTPUT / f"final_eval_{v}.csv"
            ocr = OUTPUT / f"ocr_qr_{v}.csv"
            if not final.is_file() or not ocr.is_file():
                continue
            cmd = [PY, str(SCRIPTS / "llm_product_name.py"),
                   "--final", str(final), "--ocr", str(ocr),
                   "--only-empty", "--min-confidence", "30"]
            if gguf and gguf.is_file():
                cmd.extend(["--gguf", str(gguf)])
            ok = _run(cmd, f"LLM {v} (only-empty)", check=False)
            if not ok:
                print(f"  LLM failed for {v}")

    _run(
        [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(OUTPUT),
         "--finalize-names", *cat_arg],
        "finalize product_name",
        check=not args.dry_run,
    )

    if not args.dry_run:
        SUBMISSION.mkdir(parents=True, exist_ok=True)
        for v in VIDEOS:
            src = OUTPUT / f"final_eval_{v}.csv"
            if src.is_file():
                shutil.copy2(src, SUBMISSION / f"final_eval_{v}.csv")
        _run([PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(OUTPUT)], "eval")
        print(f"\nBackup before enrich: {snap}")
    else:
        print("\n(dry-run, prod not written)")


if __name__ == "__main__":
    main()
