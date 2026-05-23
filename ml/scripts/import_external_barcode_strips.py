"""Импорт внешних barcode-датасетов → digit-strip для Slot-CNN."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from barcode_strip_common import (  # noqa: E402
    STRIP_H,
    STRIP_W,
    augment_strip,
    extract_digit_strip,
    normalize_barcode,
)

EXTERNAL = ROOT / "ml" / "data" / "external_barcode"
OUT_DIR = ROOT / "ml" / "data" / "barcode_strips_external"

EAN13_RE = re.compile(r"(\d{13})")
EAN12_RE = re.compile(r"(\d{12})")


def _ean_from_name(name: str) -> str:
    m = EAN13_RE.search(name)
    if m:
        return normalize_barcode(m.group(1))
    m = EAN12_RE.search(name)
    if m:
        return normalize_barcode(m.group(1))
    return ""


def _decode_ean_pyzbar(img: np.ndarray) -> str:
    try:
        from pyzbar.pyzbar import decode as zdecode
    except ImportError:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    for obj in zdecode(gray):
        raw = obj.data.decode("utf-8", errors="ignore")
        bc = normalize_barcode(raw)
        if len(bc) >= 12:
            return bc[:13]
    return ""


def _strip_from_barcode_image(img: np.ndarray, code: str) -> np.ndarray | None:
    """Нижняя зона под штрихом или fallback на синтетику."""
    for y0, y1 in ((0.55, 0.98), (0.65, 0.98), (0.45, 0.95)):
        strip = extract_digit_strip(img, y_start=y0, y_end=y1, pad_x=0.01)
        if strip is not None:
            return strip
    from barcode_strip_common import render_synthetic_strip

    return render_synthetic_strip(code)


def _append_row(rows: list[dict], sid: str, source: str, code: str, strip: np.ndarray, out: Path):
    rel = f"strips/{sid}.png"
    cv2.imwrite(str(out / rel), strip)
    rows.append({
        "id": sid,
        "source": source,
        "video": "",
        "barcode": code[:13],
        "strip_path": rel,
    })


def import_sbd_lr(data_dir: Path, out: Path, limit: int, aug: int, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    archive = data_dir / "BarcodesLR.tar.gz"
    extract = data_dir / "sbd_lr"
    if archive.is_file() and not extract.is_dir():
        print(f"Extracting {archive.name}…")
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract)

    img_dirs = [extract]
    img_dirs += [p for p in extract.rglob("*") if p.is_dir() and p.name.lower() in ("images", "img", "dataset")]
    images: list[Path] = []
    for d in img_dirs:
        images.extend(sorted(d.glob("*.jpg")))
        images.extend(sorted(d.glob("*.png")))
    if not images and extract.is_dir():
        images = sorted(extract.rglob("*.jpg")) + sorted(extract.rglob("*.png"))
    print(f"SBD LR: {len(images)} images under {extract}")

    n = 0
    for p in images:
        if limit and n >= limit:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        code = _decode_ean_pyzbar(img) or _ean_from_name(p.stem)
        if len(code) < 12:
            continue
        strip = _strip_from_barcode_image(img, code)
        if strip is None:
            continue
        sid = f"sbd_{n:05d}"
        _append_row(rows, sid, "sbd_lr", code, strip, out)
        for j in range(aug):
            sid_a = f"sbd_{n:05d}_a{j}"
            _append_row(rows, sid_a, "sbd_lr_aug", code, augment_strip(strip, rng), out)
        n += 1
    return rows


def import_barber_vgg(data_dir: Path, out: Path, limit: int, aug: int, rng: np.random.Generator) -> list[dict]:
    """BarBeR: папки dataset/ + Annotations/*.json (VGG)."""
    rows: list[dict] = []
    img_root = data_dir / "dataset"
    ann_root = data_dir / "Annotations"
    if not img_root.is_dir() or not ann_root.is_dir():
        print("BarBeR: skip (place dataset/ + Annotations/ under external_barcode/barber/)")
        return rows

    n = 0
    for ann_path in sorted(ann_root.glob("*.json")):
        if limit and n >= limit:
            break
        try:
            meta = json.loads(ann_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        img_key = meta.get("_via_img_metadata", {})
        img_ref = meta.get("_via_image_id_list", [None])[0]
        img_info = img_key.get(str(img_ref), {}) if img_ref is not None else {}
        fname = img_info.get("filename") or ann_path.stem + ".jpg"
        img_path = img_root / fname
        if not img_path.is_file():
            img_path = img_root / ann_path.stem / fname
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        regions = img_info.get("regions", [])
        for reg in regions:
            attrs = reg.get("region_attributes", {})
            btype = str(attrs.get("Type", attrs.get("type", ""))).upper()
            code = normalize_barcode(str(attrs.get("String", attrs.get("string", ""))))
            if "EAN" not in btype and len(code) < 12:
                continue
            if len(code) < 12:
                code = _decode_ean_pyzbar(img)
            if len(code) < 12:
                continue
            shape = reg.get("shape_attributes", {})
            xs = shape.get("all_points_x") or []
            ys = shape.get("all_points_y") or []
            if len(xs) >= 4 and len(ys) >= 4:
                x0, x1 = int(min(xs)), int(max(xs))
                y0, y1 = int(min(ys)), int(max(ys))
                pad = max(4, int((y1 - y0) * 0.35))
                y1 = min(img.shape[0], y1 + pad)
                crop = img[y0:y1, x0:x1]
            else:
                crop = img
            strip = _strip_from_barcode_image(crop, code)
            if strip is None:
                continue
            sid = f"barber_{n:05d}"
            _append_row(rows, sid, "barber", code, strip, out)
            for j in range(aug):
                _append_row(rows, f"barber_{n:05d}_a{j}", "barber_aug", code, augment_strip(strip, rng), out)
            n += 1
            if limit and n >= limit:
                break
    print(f"BarBeR: {n} EAN strips")
    return rows


def import_folder_ean13(folder: Path, out: Path, source: str, limit: int, aug: int, rng: np.random.Generator) -> list[dict]:
    """DEAL KAIST / Muenster: EAN в имени файла."""
    rows: list[dict] = []
    if not folder.is_dir():
        return rows
    images = sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")) + sorted(folder.rglob("*.JPG"))
    n = 0
    for p in images:
        if limit and n >= limit:
            break
        code = _ean_from_name(p.name)
        if len(code) < 12:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        strip = _strip_from_barcode_image(img, code)
        if strip is None:
            continue
        sid = f"{source}_{n:05d}"
        _append_row(rows, sid, source, code, strip, out)
        for j in range(aug):
            _append_row(rows, f"{source}_{n:05d}_a{j}", f"{source}_aug", code, augment_strip(strip, rng), out)
        n += 1
    print(f"{source}: {n} strips from {folder}")
    return rows


def import_zip(path: Path, out: Path, source: str, limit: int, aug: int, rng: np.random.Generator) -> list[dict]:
    if not path.is_file():
        return []
    extract = path.parent / path.stem
    if not extract.is_dir():
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract)
    return import_folder_ean13(extract, out, source, limit, aug, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=EXTERNAL)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--sbd-limit", type=int, default=3000)
    ap.add_argument("--barber-limit", type=int, default=2000)
    ap.add_argument("--folder-limit", type=int, default=1000)
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_rows: list[dict] = []
    all_rows += import_sbd_lr(args.data_dir, args.out, args.sbd_limit, args.aug, rng)
    all_rows += import_barber_vgg(args.data_dir / "barber", args.out, args.barber_limit, args.aug, rng)

    for name in ("deal_kaist", "muenster", "artelab", "inventbar"):
        folder = args.data_dir / name
        all_rows += import_folder_ean13(folder, args.out, name, args.folder_limit, args.aug, rng)
        zp = args.data_dir / f"{name}.zip"
        all_rows += import_zip(zp, args.out, name, args.folder_limit, args.aug, rng)

    labels = args.out / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "video", "barcode", "strip_path"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved {len(all_rows)} external strips → {args.out}")


if __name__ == "__main__":
    main()
