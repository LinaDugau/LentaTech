"""Zone decode + partial EAN."""
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
    print("    ", " ".join(c for c in cmd if c))
    subprocess.run([c for c in cmd if c], check=True, cwd=str(ROOT))


def _copy_prod(out_dir: Path, videos: tuple[str, ...]) -> None:
    src_out = ROOT / "ml" / "output"
    src_sub = ROOT / "ml" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in videos:
        for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
            src = src_sub / name if name.startswith("final") and (src_sub / name).is_file() else src_out / name
            if not src.is_file():
                src = src_out / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)
            else:
                print(f"  warn: missing {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=list(VIDEOS))
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--from-workdir", type=Path, default=None,
                    help="использовать готовый workdir (не копировать prod)")
    ap.add_argument("--llm-only", action="store_true",
                    help="только LLM rerank + rematch + eval (workdir уже готов)")
    ap.add_argument("--fast", action="store_true", help="быстрый decode")
    ap.add_argument("--video-decode", action="store_true",
                    help="4K decode из видео (медленно)")
    ap.add_argument("--undistort", action="store_true")
    ap.add_argument("--with-llm", action="store_true", help="LLM rerank (медленно)")
    ap.add_argument("--llm-fill-barcode", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if args.from_workdir:
        work = Path(args.from_workdir)
        if not work.is_dir():
            raise SystemExit(f"from-workdir not found: {work}")
    else:
        work = args.workdir or (ROOT / "ml" / "output" / f"_top3_{ts}")
        work.mkdir(parents=True, exist_ok=True)
        if not args.llm_only:
            _copy_prod(work, tuple(args.videos))
    print(f"Workdir: {work}")

    catalog = ROOT / "db_hack.csv"
    hybrid = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
    cat = hybrid if hybrid.is_file() else catalog

    if not args.llm_only:
        for v in args.videos:
            ocr = work / f"ocr_qr_{v}.csv"
            if not ocr.is_file():
                print(f"skip {v}: no ocr")
                continue
            _run(
                [PY, str(SCRIPTS / "partial_ean_recovery.py"),
                 "--video", v, "--ocr", str(ocr), "--catalog", str(catalog)],
                f"2/3 partial_ean_recovery {v}",
            )
            zone_cmd = [
                PY, str(SCRIPTS / "barcode_zone_decode.py"),
                "--files", str(ocr), "--catalog", str(catalog),
                "--top-k", "3", "--fast",
            ]
            if not args.video_decode:
                zone_cmd.append("--crop-only")
            if args.undistort:
                zone_cmd.append("--undistort")
            _run(zone_cmd, f"1/3 barcode_zone_decode {v}")

        _run(
            [PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
             "--final-dir", str(work), "--ocr-dir", str(work)],
            "apply_qr_to_final",
        )

        for v in args.videos:
            final = work / f"final_eval_{v}.csv"
            if not final.is_file():
                continue
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

    if args.with_llm or args.llm_only:
        for v in args.videos:
            final = work / f"final_eval_{v}.csv"
            ocr = work / f"ocr_qr_{v}.csv"
            if not final.is_file():
                continue
            cmd = [
                PY, str(SCRIPTS / "llm_catalog_rerank.py"),
                "--final", str(final), "--ocr", str(ocr), "--catalog", str(cat),
            ]
            if args.llm_fill_barcode:
                cmd.append("--fill-barcode")
            _run(cmd, f"3/3 llm_catalog_rerank {v}")

        for v in args.videos:
            final = work / f"final_eval_{v}.csv"
            if not final.is_file():
                continue
            _run(
                [PY, str(SCRIPTS / "match_to_catalog.py"),
                 "--final", str(final), "--catalog", str(cat),
                 "--threshold", "0.70", "--overwrite"],
                f"match_to_catalog post-LLM {v}",
            )
        _run(
            [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
            "final_postprocess (post-LLM)",
        )

    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "evaluate_official",
    )
    print(f"\nDone. Results in {work}")


if __name__ == "__main__":
    main()
