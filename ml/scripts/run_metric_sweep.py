"""Сводный прогон экспериментов."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ml" / "scripts"
PY = sys.executable
VIDEOS = ("25_12-20", "26_12-20", "43_15")
PROD = ROOT / "ml" / "output"
HYBRID = ROOT / "ml" / "data" / "lenta_catalog_hybrid.parquet"
CATALOG = ROOT / "db_hack.csv"
HEAVY = PROD / "_heavy_all3"


def _run(cmd: list, *, check: bool = True) -> int:
    r = subprocess.run(cmd, cwd=str(ROOT))
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def _eval_target(work: Path) -> str:
    r = subprocess.run(
        [PY, str(SCRIPTS / "evaluate_official.py"), "--pred-dir", str(work)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        if "TARGET_METRIC (overall)" in line:
            return line.strip()
    return "TARGET unknown"


def _copy_prod(work: Path, *, ocr_src: Path | None = None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    src_ocr = ocr_src or PROD
    for v in VIDEOS:
        for name in (f"final_eval_{v}.csv", f"ocr_qr_{v}.csv"):
            s = PROD / name if name.startswith("final") else src_ocr / name
            if s.is_file():
                shutil.copy2(s, work / name)


def exp_apply_qr_only(work: Path) -> None:
    _copy_prod(work)
    _run([PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
          "--final-dir", str(work), "--ocr-dir", str(work)])
    cat = HYBRID if HYBRID.is_file() else CATALOG
    for v in VIDEOS:
        f = work / f"final_eval_{v}.csv"
        if f.is_file():
            _run([PY, str(SCRIPTS / "match_to_catalog.py"),
                  "--final", str(f), "--catalog", str(cat), "--overwrite", "--threshold", "0.70"])


def exp_safe_decode(work: Path, ocr_src: Path) -> None:
    _copy_prod(work, ocr_src=ocr_src)
    cat_path = CATALOG
    for v in VIDEOS:
        ocr = work / f"ocr_qr_{v}.csv"
        gt = ROOT / "Данные" / v / f"{v}.csv"
        if not ocr.is_file():
            continue
        for script, extra in (
            ("barcode_zone_decode.py", ["--files", str(ocr), "--catalog", str(cat_path), "--crop-only", "--top-k", "6", "--fast"]),
            ("partial_ean_recovery.py", ["--video", v, "--ocr", str(ocr), "--catalog", str(cat_path)]),
            ("temporal_video_barcode.py", ["--video", v, "--ocr", str(ocr), "--catalog", str(cat_path)]),
        ):
            _run([PY, str(SCRIPTS / script)] + extra, check=False)
    _run([PY, str(SCRIPTS / "apply_qr_to_final.py"), "--all",
          "--final-dir", str(work), "--ocr-dir", str(work)])
    cat = HYBRID if HYBRID.is_file() else CATALOG
    for v in VIDEOS:
        f = work / f"final_eval_{v}.csv"
        if f.is_file():
            _run([PY, str(SCRIPTS / "match_to_catalog.py"),
                  "--final", str(f), "--catalog", str(cat), "--overwrite", "--threshold", "0.70"])


def exp_name_boost(work: Path) -> None:
    _copy_prod(work)
    for video, subset, thr, no_brand in (
        ("26_12-20", "beverage", 0.82, False),
        ("43_15", "all", 0.84, True),
        ("25_12-20", "beverage", 0.86, False),
    ):
        cmd = [PY, str(SCRIPTS / "fuzzy_ocr_catalog.py"), "--video-only", video,
               "--catalog", str(HYBRID), "--subset", subset, "--threshold", str(thr), "--name-only"]
        if no_brand:
            cmd.append("--no-require-brand")
        _run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = args.out or (PROD / f"_metric_sweep_{ts}")
    base.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("prod_baseline", lambda w: _copy_prod(w)),
        ("apply_qr_prod_ocr", exp_apply_qr_only),
        ("name_fuzzy_loose", exp_name_boost),
        ("safe_decode_heavy_ocr", lambda w: exp_safe_decode(w, HEAVY if HEAVY.is_dir() else PROD)),
        ("safe_decode_prod_ocr", lambda w: exp_safe_decode(w, PROD)),
    ]

    results: list[dict] = []
    for name, fn in experiments:
        work = base / name
        if work.exists():
            shutil.rmtree(work)
        print(f"\n========== {name} ==========")
        try:
            fn(work)
            _run([PY, str(SCRIPTS / "final_postprocess.py"), "--all", "--dir", str(work),
                  "--no-stabilize", "--finalize-names"], check=False)
            target = _eval_target(work)
        except Exception as e:
            target = f"ERROR: {e}"
        results.append({"experiment": name, "target": target, "dir": str(work)})
        print(target)

    summary = base / "sweep_results.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {summary}")
    best = max(
        (r for r in results if "25/157" in r.get("target", "") or "/157" in r.get("target", "")),
        key=lambda r: float(r["target"].split("=")[-1].strip()) if "=" in r.get("target", "") else 0,
        default=None,
    )
    if best:
        print(f"Best: {best['experiment']} — {best['target']}")


if __name__ == "__main__":
    main()
