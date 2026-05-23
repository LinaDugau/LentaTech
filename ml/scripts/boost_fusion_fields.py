"""Fusion id_sku, code, datetime."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable
VIDEOS = ("25_12-20", "26_12-20", "43_15")


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
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
    ap.add_argument("--workdir", type=Path, default=ROOT / "ml" / "output" / "_day20_fusion")
    ap.add_argument("--from-workdir", type=Path, default=None)
    ap.add_argument("--bottom-ocr", action="store_true", help="extra_bottom_ocr per-track sharpest")
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()

    work = Path(args.from_workdir) if args.from_workdir else Path(args.workdir)
    if not args.from_workdir and not args.skip_copy:
        _copy_prod(work, tuple(args.videos))
    print(f"Workdir: {work}")

    if args.bottom_ocr:
        for v in args.videos:
            ocr = work / f"ocr_qr_{v}.csv"
            final = work / f"final_eval_{v}.csv"
            if not ocr.is_file():
                continue
            cmd = [
                PY, str(SCRIPTS / "extra_bottom_ocr.py"),
                "--files", str(ocr),
                "--per-track", "--sharpest", "--force",
                "--source", "hires",
            ]
            if final.is_file():
                cmd.extend(["--final", str(final)])
            _run(cmd, f"bottom OCR {v}")

    for v in args.videos:
        final = work / f"final_eval_{v}.csv"
        ocr = work / f"ocr_qr_{v}.csv"
        if final.is_file() and ocr.is_file():
            _run(
                [
                    PY, str(SCRIPTS / "patch_final_from_ocr.py"),
                    "--video", v, "--dir", str(work),
                ],
                f"patch_final {v}",
            )

    _run(
        [PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work)],
        "final_postprocess (fusion + reparse)",
    )
    _run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        "evaluate_official",
    )
    print(f"\nDone. Results in {work}")


if __name__ == "__main__":
    main()
