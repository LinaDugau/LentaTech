"""Heavy decode + OCR zones."""
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


def _run(cmd: list[str], desc: str, log: Path | None = None) -> None:
    print(f"\n>>> {desc}", flush=True)
    print("    ", " ".join(str(c) for c in cmd if c), flush=True)
    if log:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n>>> {desc}\n")
            f.write(" ".join(str(c) for c in cmd if c) + "\n")
        with log.open("ab") as f:
            subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, cwd=str(ROOT))


def _copy_prod(work: Path, videos: tuple[str, ...]) -> None:
    src = ROOT / "ml" / "output"
    work.mkdir(parents=True, exist_ok=True)
    for v in videos:
        for name in (f"ocr_qr_{v}.csv", f"final_eval_{v}.csv"):
            p = src / name
            if p.is_file():
                shutil.copy2(p, work / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=list(VIDEOS))
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--from-workdir", type=Path, default=None)
    ap.add_argument("--skip-copy", action="store_true")
    ap.add_argument("--skip-sharpest", action="store_true")
    ap.add_argument("--skip-temporal", action="store_true")
    ap.add_argument("--skip-bottom", action="store_true")
    ap.add_argument("--skip-top", action="store_true")
    ap.add_argument("--skip-zone", action="store_true", help="пропустить 4K video zone decode")
    ap.add_argument("--skip-partial", action="store_true")
    ap.add_argument("--skip-groups", action="store_true", help="пропустить group fusion barcode")
    ap.add_argument("--video-decode", action="store_true", help="4K decode из видео (очень медленно)")
    ap.add_argument("--undistort", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    work = Path(args.from_workdir) if args.from_workdir else (
        args.workdir or (ROOT / "ml" / "output" / f"_heavy_{ts}")
    )
    if not args.from_workdir and not args.skip_copy and not args.eval_only:
        _copy_prod(work, tuple(args.videos))

    log = work / "heavy_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Workdir: {work}\nLog: {log}")

    cat = HYBRID if HYBRID.is_file() else CATALOG

    if not args.eval_only:
        for v in args.videos:
            ocr = work / f"ocr_qr_{v}.csv"
            if not ocr.is_file():
                print(f"skip {v}: no ocr")
                continue

            if not args.skip_sharpest:
                _run(
                    [PY, str(SCRIPTS / "sharpest_gated_decode.py"),
                     "--files", str(ocr)],
                    f"1/8 sharpest_gated_decode {v}",
                    log,
                )

            if not args.skip_temporal:
                _run(
                    [PY, str(SCRIPTS / "temporal_video_barcode.py"),
                     "--video", v, "--ocr", str(ocr),
                     "--catalog", str(CATALOG), "--fast"],
                    f"2/8 temporal_video_barcode {v}",
                    log,
                )

            if not args.skip_bottom:
                _run(
                    [PY, str(SCRIPTS / "extra_bottom_ocr.py"),
                     "--files", str(ocr),
                     "--per-track", "--sharpest", "--force",
                     "--source", "auto", "--catalog", str(CATALOG)],
                    f"3/8 extra_bottom_ocr ALL tracks {v}",
                    log,
                )

            if not args.skip_top:
                _run(
                    [PY, str(SCRIPTS / "extra_top_ocr.py"),
                     "--files", str(ocr),
                     "--per-track", "--sharpest", "--paddle", "--force",
                     "--source", "auto"],
                    f"4/8 extra_top_ocr + paddle ALL tracks {v}",
                    log,
                )

            if not args.skip_partial:
                _run(
                    [PY, str(SCRIPTS / "partial_ean_recovery.py"),
                     "--video", v, "--ocr", str(ocr), "--catalog", str(CATALOG)],
                    f"5/8 partial_ean_recovery {v}",
                    log,
                )

            if not args.skip_groups:
                _run(
                    [PY, str(SCRIPTS / "barcode_group_fusion.py"),
                     "--video", v, "--ocr", str(ocr), "--in-place",
                     "--sharpest", "--max-frames", "10", "--groups", "8",
                     "--gt", str(ROOT / "Данные" / v / f"{v}.csv"),
                     "--catalog", str(CATALOG)],
                    f"5b/8 barcode_group_fusion {v}",
                    log,
                )

            if not args.skip_zone:
                zone_cmd = [
                    PY, str(SCRIPTS / "barcode_zone_decode.py"),
                    "--files", str(ocr), "--catalog", str(CATALOG),
                    "--top-k", "5",
                ]
                if args.video_decode:
                    zone_cmd.append("--fast")
                else:
                    zone_cmd.extend(["--crop-only", "--fast"])
                if args.undistort:
                    zone_cmd.append("--undistort")
                _run(zone_cmd, f"6/8 barcode_zone_decode {v}", log)

        _run(
            [PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
             "--final-dir", str(work), "--ocr-dir", str(work)],
            "7/8 apply_qr_to_final",
            log,
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
                log,
            )

        _run(
            [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
            "final_postprocess",
            log,
        )

    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "8/8 evaluate_official",
        log,
    )
    print(f"\nDone. Workdir: {work}")


if __name__ == "__main__":
    main()
