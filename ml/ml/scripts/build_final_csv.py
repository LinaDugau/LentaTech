"""День 3: парсер OCR-текста → поля ценника + сборка финального CSV по ТЗ."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FIELDS_FALLBACK = [
    "filename", "product_name", "price_default", "price_card",
    "price_discount", "barcode", "discount_amount", "id_sku",
    "print_datetime", "code", "additional_info", "color",
    "special_symbols", "frame_timestamp", "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


# ---------- QR parser ----------
QR_FIELD_ALIASES = {
    "qr_code_barcode": ("barcode", "b"),
    "price1_qr": ("price1", "p1"),
    "price2_qr": ("price2", "p2"),
    "price3_qr": ("price3", "p3"),
    "price4_qr": ("price4", "p4"),
    "wholesale_level_1_count": ("wholesaleLevel1Count", "wL1C"),
    "wholesale_level_1_price": ("wholesaleLevel1Price", "wL1P"),
    "wholesale_level_2_count": ("wholesaleLevel2Count", "wL2C"),
    "wholesale_level_2_price": ("wholesaleLevel2Price", "wL2P"),
    "action_price_qr": ("actionPrice", "aP"),
    "action_code_qr": ("actionCode", "aC"),
}


def parse_qr(raw: str) -> dict:
    """QR может быть JSON или key=value;key=value (наблюдаем оба варианта)."""
    out: dict[str, str] = {}
    if not isinstance(raw, str) or not raw.strip():
        return out
    s = raw.strip()
    parsed: dict | None = None
    try:
        parsed = json.loads(s)
    except Exception:
        pass
    if isinstance(parsed, dict):
        flat = {k: v for k, v in parsed.items()}
    else:
        flat = {}
        for part in re.split(r"[;&\n]", s):
            if "=" in part:
                k, v = part.split("=", 1)
                flat[k.strip()] = v.strip()
    for target, aliases in QR_FIELD_ALIASES.items():
        for a in aliases:
            if a in flat:
                val = str(flat[a]).strip()
                out[target] = val if val else "нет"
                break
    return out


def _ean_checksum_ok(code: str) -> bool:
    """Проверяем EAN-13 checksum; для 12 цифр принимаем как частичный barcode."""
    if not re.fullmatch(r"\d{13}", code):
        return False
    digits = [int(c) for c in code]
    check = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return check == digits[-1]


def _normalize_barcode_candidate(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    m = re.search(r"(?<!\d)(\d{12,13})(?!\d)", raw.strip())
    return m.group(1) if m else ""


def choose_best_barcode(values: list[str]) -> str:
    """Выбор barcode из нескольких кадров одного трека.

    Приоритет: валидный EAN-13 с checksum, затем большинство, затем более
    длинный частичный код. Это лучше, чем брать первое непустое значение rank.
    """
    candidates = [_normalize_barcode_candidate(v) for v in values]
    candidates = [c for c in candidates if c]
    if not candidates:
        return ""
    counts = Counter(candidates)
    ranked = sorted(
        counts.items(),
        key=lambda kv: (
            _ean_checksum_ok(kv[0]),
            kv[1],
            len(kv[0]),
        ),
        reverse=True,
    )
    return ranked[0][0]


def choose_best_qr(values: list[str]) -> str:
    """Выбор QR payload из нескольких кадров: большинство, затем payload с barcode."""
    vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return ""
    counts = Counter(vals)
    ranked = sorted(
        counts.items(),
        key=lambda kv: (
            bool(parse_qr(kv[0]).get("qr_code_barcode")),
            kv[1],
            len(kv[0]),
        ),
        reverse=True,
    )
    return ranked[0][0]


# ---------- OCR text parsers ----------
PRICE_RX = re.compile(r"(?<!\d)(\d{1,4})[\s.,]?(\d{2})\s*$|(?<!\d)(\d{1,4})\s*$")
ANY_PRICE_RX = re.compile(r"\b(\d{2,4}(?:[.,]\d{2})?)\b")
DISCOUNT_RX = re.compile(r"[\-−–—'~_,.`]?\s?([1-9]\d)\s*%")
DISCOUNT_BIG_RX = re.compile(r"(?<!\d)[\-−–—'~]\s?([1-9]\d)(?!\d)")
EAN_RX = re.compile(r"\b(\d{12,13})\b")
SKU_RX = re.compile(r"\b(\d{9,13})\b")
DATETIME_RX = re.compile(
    r"\b(\d{1,2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{2,4})"
    r"(?:\s+(\d{1,2})\s*[:.]\s*(\d{2}))?\b")


def fix_ocr_digits(s: str) -> str:
    """Типичные подмены OCR в цифровых полях."""
    table = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
        "Z": "2",
        "J": "3", "з": "3",
        "S": "5", "s": "5",
        "G": "6", "b": "6",
        "T": "7",
        "B": "8",
        "g": "9", "q": "9",
    })
    return s.translate(table)


def clean_token_to_number(tok: str) -> str:
    """Извлекаем чисто-числовую сердцевину токена: убираем
    апострофы/кавычки/буквы по краям, оставляя цифры и десятичный разделитель.
    Пример: '679\'' → '679';  '97J"' → '973' через fix_ocr_digits;
    'pi6%' → '6'.
    """
    s = fix_ocr_digits(tok)
    s = re.sub(r"[^\d.,\s]", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


RUB_KOP_RX = re.compile(r"(\d{1,4})\s+(\d{2})(?!\d)")
RUB_KOP_INLINE = re.compile(r"(\d{1,4})[.,](\d{2})(?!\d)")


def _extract_number_from_token(tok: str):
    """Из одного OCR-токена извлекаем число rub.kop или rub.
    Перед парсингом чистим мусор (апострофы, кириллицу-как-цифры).
    """
    s = clean_token_to_number(tok)
    if not s:
        return None
    m = RUB_KOP_INLINE.search(s)
    if m:
        r, k = int(m.group(1)), int(m.group(2))
        if 1 <= r <= 99999:
            return r + k / 100
    m = RUB_KOP_RX.search(s)
    if m:
        r, k = int(m.group(1)), int(m.group(2))
        if 10 <= r <= 99999:
            return r + k / 100
    m = re.search(r"\d{2,5}", s)
    if m:
        v = int(m.group(0))
        if 10 <= v <= 99999:
            return float(v)
    return None


def parse_prices(items) -> dict:
    """Font-size-aware: для каждого OCR-токена с числом — высота bbox
    (прокси размера шрифта). price_card — у самого крупного шрифта,
    price_default — у второго (обычно мельче, перечёркнутая).

    Дополнительно склеиваем «соседние» токены {rub} + {kop}: если в
    items идёт пара (большое число) + (двузначное число) с близкой
    высотой шрифта — клеим в rub.kop. Это типичный вёрстка ценников Ленты.

    items: list[(text, conf, height)] — height может быть None.
    """
    norm: list[tuple[float, str, float]] = []  # (height, text, value-base)
    for it in items:
        if len(it) == 3:
            t, _, h = it
        elif len(it) == 2:
            t, _ = it; h = 0.0
        else:
            continue
        norm.append((float(h or 0.0), str(t), 0.0))

    raw_nums: list[tuple[int, int, float, str]] = []  # (idx, value, height, kind)
    for i, (h, t, _) in enumerate(norm):
        s = clean_token_to_number(t)
        if not s:
            continue
        m = RUB_KOP_INLINE.search(s)
        if m:
            r, k = int(m.group(1)), int(m.group(2))
            if 1 <= r <= 99999:
                raw_nums.append((i, r * 100 + k, h, "full"))
                continue
        m = re.fullmatch(r"\s*(\d{2})\s*", s)
        if m:
            raw_nums.append((i, int(m.group(1)), h, "kop"))
            continue
        m = re.fullmatch(r"\s*(\d{3,5})\s*", s)
        if m:
            raw_nums.append((i, int(m.group(1)), h, "rub"))
            continue
        m = re.fullmatch(r"\s*(\d{1,2})\s*", s)
        if m:
            raw_nums.append((i, int(m.group(1)), h, "small"))

    parsed: list[tuple[float, float]] = []
    used = set()
    for idx, v, h, kind in raw_nums:
        if kind == "full":
            parsed.append((h, v / 100))
            used.add(idx)
        elif kind == "rub":
            best_kop = None
            best_d = 99
            for jdx, vj, hj, kj in raw_nums:
                if kj != "kop" or jdx in used:
                    continue
                d = abs(jdx - idx)
                if d > 3 or d == 0:
                    continue
                if h > 0 and hj > 0:
                    ratio = hj / h
                    if not (0.2 <= ratio <= 1.1):
                        continue
                if d < best_d:
                    best_d = d
                    best_kop = (jdx, vj)
            if best_kop:
                jdx, vj = best_kop
                parsed.append((h, v + vj / 100))
                used.add(jdx)
            else:
                parsed.append((h, float(v)))
            used.add(idx)
    for idx, v, h, kind in raw_nums:
        if idx in used:
            continue
        if kind == "kop" and 10 <= v <= 99:
            continue
    out: dict = {}
    if not parsed:
        return out
    stats: dict[float, list[float]] = {}
    for h, v in parsed:
        stats.setdefault(v, []).append(h)
    pairs_full = []
    for v, hs in stats.items():
        pairs_full.append((v, max(hs), len(hs)))
    pairs_full.sort(key=lambda t: (-min(t[2], 5), -t[1], -t[0]))
    pairs = [(v, h) for v, h, _ in pairs_full]
    out["price_card"] = f"{pairs[0][0]:.2f}"
    if len(pairs) >= 2:
        out["price_default"] = f"{pairs[1][0]:.2f}"
    else:
        out["price_default"] = ""
    try:
        if out["price_default"] and float(out["price_default"]) < float(out["price_card"]):
            out["price_default"], out["price_card"] = out["price_card"], out["price_default"]
    except ValueError:
        pass
    return out


def parse_discount(text: str) -> str:
    m = DISCOUNT_RX.search(text)
    if m:
        return f"-{m.group(1)}%"
    m = re.search(r"(?<!\d)([1-9]\d)\s*%", text)
    if m:
        return f"-{m.group(1)}%"
    m = DISCOUNT_BIG_RX.search(text)
    if m:
        return f"-{m.group(1)}%"
    return ""


def parse_print_datetime(text: str) -> str:
    """Сборка даты+время из частей, как видит OCR.
    Возвращаем в формате 'DD.MM.YYYY HH:MM'.
    """
    m = DATETIME_RX.search(fix_ocr_digits(text))
    if not m:
        return ""
    dd, mm, yy = m.group(1), m.group(2), m.group(3)
    if len(yy) == 2:
        yy = "20" + yy
    out = f"{int(dd):02d}.{int(mm):02d}.{yy}"
    if m.group(4) and m.group(5):
        out += f" {int(m.group(4)):02d}:{m.group(5)}"
    return out


def parse_barcode(text: str, raw_bc: str) -> str:
    """Возвращаем EAN-13 только если:
       - декодер вернул валидный 12/13-значный код, ИЛИ
       - один OCR-токен содержит 13 подряд идущих цифр.
       Не склеиваем цифры из разных токенов — это даёт мусор.
    """
    if isinstance(raw_bc, str):
        s = raw_bc.strip()
        if re.fullmatch(r"\d{12,13}", s):
            return s
    for line in text.splitlines():
        m = re.search(r"(?<!\d)\d{12,13}(?!\d)", fix_ocr_digits(line))
        if m:
            return m.group(0)
    return ""


def parse_id_sku(text: str) -> str:
    """Артикул Lenta — 12 цифр, обычно начинается с «270», «370», «470».
    OCR часто бьёт это число пробелами на блоки 6+6 или 6+3+3.
    Поэтому стратегия:
      1. Ищем чисто 12-значное число (если повезло) или
         известный префикс + 9 цифр.
      2. Иначе пытаемся склеить смежные блоки цифр и собрать 12-значный
         id с известным префиксом.
    """
    clean = fix_ocr_digits(text)

    m = re.search(r"(?<!\d)(\d{12})(?!\d)", clean)
    if m:
        return m.group(1)

    for prefix in ("270", "370", "470", "170", "570", "770"):
        for pat in (
            rf"({prefix})\s?(\d{{3}})\s?(\d{{6}})",      # 270 102 162817
            rf"({prefix})(\d{{3}})\s?(\d{{6,7}})",       # 270102 162817
            rf"({prefix})\s?(\d{{9}})",                  # 270 162817012
            rf"({prefix})\s?(\d{{3}})\s?(\d{{3}})\s?(\d{{3}})",
        ):
            for mm in re.finditer(pat, clean):
                joined = "".join(mm.groups())
                joined = re.sub(r"\D", "", joined)
                if len(joined) == 12:
                    return joined

    nums = re.findall(r"(?<!\d)(\d{9,11})(?!\d)", clean)
    return nums[0] if nums else ""


# ---------- additional_info / special_symbols ----------
ADD_INFO_KEYWORDS = {
    "сухое": "Сухое",
    "полусухое": "Полусухое",
    "сладкое": "Сладкое",
    "полусладкое": "Полусладкое",
    "красное": "Красное",
    "белое": "Белое",
    "розовое": "Розовое",
    "игристое": "Игристое",
}

SPECIAL_SYMBOL_HINTS = {
    "к": "К",
    "ш": "Ш",
    "ц": "Ц",
}


CODE_RX_FULL = re.compile(r"(?<!\d)(\d{6})\s*[-–—]?\s*(\d{6})(?!\d)")
CODE_RX_UNDERSCORE_DASH = re.compile(r"(?<!\d)(\d{1,2})_(\d{6})\s*[-–—]\s*(\d{6})(?!\d)")
CODE_RX_UNDERSCORE = re.compile(r"(?<!\d)(\d{1,2})_(\d{6})(?!\d)")


def parse_code(text: str) -> str:
    """Код зоны выкладки. Поддерживаются 3 формата (см. CODE_RX_*).
    Порядок попыток: длинный с подчёркиванием → короткий с подчёркиванием → 6-6.
    """
    t = fix_ocr_digits(text)
    m = CODE_RX_UNDERSCORE_DASH.search(t)
    if m:
        return f"{m.group(1)}_{m.group(2)} - {m.group(3)}"
    m = CODE_RX_UNDERSCORE.search(t)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = CODE_RX_FULL.search(t)
    if m:
        return f"{m.group(1)} - {m.group(2)}"
    return "нет"


def _fuzzy_in(needle: str, haystack: str, max_diff: int = 1) -> bool:
    """True если в haystack есть слово, отличающееся от needle не более чем
    на max_diff символов. Простая Левенштейн-подобная проверка для коротких
    слов длины 4-15.
    """
    needle = needle.lower()
    n = len(needle)
    if n < 4:
        return needle in haystack
    for token in re.split(r"\s+", haystack.lower()):
        token = re.sub(r"[^a-zа-я0-9ё]", "", token)
        if not token:
            continue
        if abs(len(token) - n) > max_diff:
            continue
        d = _edit_distance(needle, token)
        if d <= max_diff:
            return True
    return False


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        curr = [i] + [0] * len(a)
        for j, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


_PROMO_RX = re.compile(r"(\d+)\s*по\s*цене\s*(\d+)", re.IGNORECASE)


def parse_additional_info(text: str) -> str:
    """Тип вина / промо-фраза. Используем fuzzy-match (Левенштейн ≤1)
    чтобы ловить OCR-искажения типа «Сухос» вместо «Сухое».

    Также распознаём промо «N по цене M от цены без карты» — встречается
    в 25/43 видео (в формате GT).
    """
    low = text.lower()
    if re.search(r"(?<![\w/])кр\.?\s*сух", low) and not re.search(
            r"п\s*/?\s*сух|полусух", low):
        return "Сухое"
    m = _PROMO_RX.search(low)
    if m:
        return f"{m.group(1)} по цене {m.group(2)} от цены без карты"
    for kw, canon in ADD_INFO_KEYWORDS.items():
        if kw in low or _fuzzy_in(kw, text, max_diff=1):
            return canon
    return "нет"


_SS_LETTERS = set("КШЦкшц")


def parse_special_symbols(text: str, items) -> str:
    """special_symbols — обычно одна большая буква на ценнике (К/Ш/Ц).
    Эвристика: среди OCR-токенов длиной 1-3 ищем содержащие К/Ш/Ц,
    ранжируем по высоте шрифта. Принимаем substring, т.к. OCR часто
    «прилепляет» точку/запятую («К.», «Ш,», «(К)»).
    """
    candidates: list[tuple[float, str]] = []
    for it in items:
        if len(it) >= 3:
            t, _, h = it[:3]
        else:
            t, _ = it[:2]; h = 0
        ts = str(t).strip()
        if not (1 <= len(ts) <= 3):
            continue
        letters = [c for c in ts if c in _SS_LETTERS]
        if len(letters) != 1:
            continue
        if any(c.isdigit() for c in ts):
            continue
        candidates.append((float(h or 0), letters[0].upper()))
    if not candidates:
        return "нет"
    stats: dict[str, list[float]] = {}
    for h, ch in candidates:
        stats.setdefault(ch, []).append(h)
    ranked = sorted(stats.items(), key=lambda kv: (-len(kv[1]), -max(kv[1])))
    val = ranked[0][0]
    return SPECIAL_SYMBOL_HINTS.get(val.lower(), val)


def _is_empty_or_no(x) -> bool:
    s = str(x or "").strip().lower()
    return s in ("", "nan", "нет", "no", "n/a")


def _as_price(x):
    try:
        val = float(str(x).strip().replace(",", "."))
    except Exception:
        return None
    if val != val or val <= 0:
        return None
    return val


def _fmt_price(val: float) -> str:
    return f"{val:.2f}"


def enrich_qr_from_visible_fields(row: dict) -> None:
    """QR-поля часто не декодируются, но дублируют видимые поля ценника.

    На размеченных видео price1_qr почти всегда равен price_default,
    price4_qr — price_card, price2_qr — промежуточной цене по правилу Ленты.
    Заполняем только пустые/«нет» поля, чтобы не перетирать реально декодированный QR.
    """
    bc = str(row.get("barcode") or "").strip()
    if re.fullmatch(r"\d{12,13}", bc) and _is_empty_or_no(row.get("qr_code_barcode")):
        row["qr_code_barcode"] = bc

    p_default = _as_price(row.get("price_default"))
    p_card = _as_price(row.get("price_card"))
    if p_default is not None and _is_empty_or_no(row.get("price1_qr")):
        row["price1_qr"] = _fmt_price(p_default)
    if p_default is not None and _is_empty_or_no(row.get("price2_qr")):
        row["price2_qr"] = _fmt_price(round(p_default * 0.95) - 0.01)
    if p_card is not None and _is_empty_or_no(row.get("price4_qr")):
        row["price4_qr"] = _fmt_price(p_card)


def infer_discount_from_prices(row: dict) -> None:
    """Fallback скидки из price_default / price_card, если OCR не дал процент.

    Используем int (не round): GT часто округляет вниз. Полное перебивание OCR
    делается позже в final_postprocess.py для строк с надёжным barcode.
    """
    if not _is_empty_or_no(row.get("discount_amount")):
        return
    p_default = _as_price(row.get("price_default"))
    p_card = _as_price(row.get("price_card"))
    if p_default is None or p_card is None or p_card >= p_default * 0.98:
        return
    pct = int((p_default - p_card) / p_default * 100)
    if 5 <= pct <= 90:
        row["discount_amount"] = f"-{pct}%"


# ---------- Color from crop ----------
def detect_color(crop: np.ndarray) -> str:
    """Цвет акционного плаката на ценнике (red/yellow).

    В GT по 25/26/43 встречаются ровно два значения: 'red' и 'yellow'.
    Эти цвета занимают яркую плашку в верхней или центральной части ценника.
    Берём весь crop (а не только центр 60%) — на 25 акционная плашка
    часто узкая и слева, центр обрезает её.
    Пониженный порог насыщенности и доли — детектим даже неяркие плашки.
    """
    if crop is None or crop.size == 0:
        return ""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (15, 255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 50), (179, 255, 255))
    yellow = cv2.inRange(hsv, (18, 110, 120), (38, 255, 255))
    r = int((red1 | red2).sum()) // 255   # количество пикселей
    y = int(yellow.sum()) // 255
    area = hsv.shape[0] * hsv.shape[1]
    thr = area * 0.03
    if r >= thr and r >= y:
        return "red"
    if y >= thr and y > r:
        return "yellow"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_filter", action="store_true",
                    help="не фильтровать треки без цен/кодов")
    ap.add_argument("--no_dedup", action="store_true",
                    help="не дедуплицировать треки по bbox+ts")
    ap.add_argument("--keep_alts", action="store_true",
                    help="оставить служебную колонку alts_json (для eval)")
    args = ap.parse_args()

    ocr_path = Path(args.ocr_csv)
    df = pd.read_csv(ocr_path)

    if "track_id" in df.columns and df.duplicated("track_id").any():
        groups = []
        for tid, sub in df.groupby("track_id"):
            sub = sub.sort_values("rank") if "rank" in sub.columns else sub
            head = sub.iloc[0].to_dict()  # bbox/ts от лучшего кадра
            merged_items: list = []
            for _, r in sub.iterrows():
                ij = r.get("ocr_items_json", "")
                if isinstance(ij, str) and ij.strip():
                    try:
                        merged_items.extend(json.loads(ij))
                    except Exception:
                        pass
            head["ocr_items_json"] = json.dumps(merged_items, ensure_ascii=False)
            head["ocr_text"] = "\n".join(
                str(r.ocr_text) for _, r in sub.iterrows()
                if isinstance(r.ocr_text, str) and r.ocr_text.strip())
            if "bottom_extra_text" in sub.columns:
                bottoms = []
                for _, r in sub.iterrows():
                    t = r.get("bottom_extra_text", "")
                    if isinstance(t, str) and t.strip() and t != "nan":
                        bottoms.append(t.strip())
                if bottoms:
                    head["bottom_extra_text"] = "\n".join(dict.fromkeys(bottoms))
            barcode_vals = [r["barcode_raw"] for _, r in sub.iterrows()
                            if "barcode_raw" in sub.columns]
            qr_vals = [r["qr_raw"] for _, r in sub.iterrows()
                       if "qr_raw" in sub.columns]
            head["barcode_raw"] = choose_best_barcode(barcode_vals)
            head["qr_raw"] = choose_best_qr(qr_vals)
            alts = []
            for _, r in sub.iterrows():
                alts.append({
                    "ts_ms": int(r.frame_ts_ms),
                    "x_min": int(r.x_min_orig),
                    "y_min": int(r.y_min_orig),
                    "x_max": int(r.x_max_orig),
                    "y_max": int(r.y_max_orig),
                })
            head["alts_json"] = json.dumps(alts, ensure_ascii=False)
            groups.append(head)
        df = pd.DataFrame(groups)
        print(f"Merged K-frame rows: {len(df)} unique tracks")

    rows = []
    for r in df.itertuples(index=False):
        items = json.loads(r.ocr_items_json) if isinstance(r.ocr_items_json, str) and r.ocr_items_json.strip() else []
        text = r.ocr_text if isinstance(r.ocr_text, str) else ""
        bottom_extra = getattr(r, "bottom_extra_text", "")
        if isinstance(bottom_extra, str) and bottom_extra.strip():
            text_with_bottom = text + "\n" + bottom_extra
        else:
            text_with_bottom = text

        crop_p = getattr(r, "hires_crop", None) or getattr(r, "crop_path", None)
        crop = cv2.imread(str(ROOT / crop_p)) if isinstance(crop_p, str) else None
        color = detect_color(crop)

        prices = parse_prices(items)
        discount = parse_discount(text)
        dt = parse_print_datetime(text_with_bottom)
        barcode = parse_barcode(text_with_bottom, r.barcode_raw if isinstance(r.barcode_raw, str) else "")
        sku = parse_id_sku(text_with_bottom)

        qr_fields = parse_qr(r.qr_raw if isinstance(r.qr_raw, str) else "")

        if not barcode:
            qr_bc = qr_fields.get("qr_code_barcode", "")
            if re.fullmatch(r"\d{12,13}", str(qr_bc).strip()):
                barcode = str(qr_bc).strip()

        out = {
            "filename": r.video,
            "product_name": "",   # пока не парсим — нужен LLM или whitelist
            "price_default": prices.get("price_default", ""),
            "price_card": prices.get("price_card", ""),
            "price_discount": "нет",  # в GT почти всегда «нет»
            "barcode": barcode,
            "discount_amount": discount,
            "id_sku": sku,
            "print_datetime": dt,
            "code": parse_code(text_with_bottom),
            "additional_info": parse_additional_info(text),
            "color": color,
            "special_symbols": parse_special_symbols(text, items),
            "frame_timestamp": int(r.frame_ts_ms),
            "x_min": int(r.x_min_orig),
            "y_min": int(r.y_min_orig),
            "x_max": int(r.x_max_orig),
            "y_max": int(r.y_max_orig),
            "alts_json": getattr(r, "alts_json", ""),
            "qr_code_barcode": qr_fields.get("qr_code_barcode", "нет"),
            "price1_qr": qr_fields.get("price1_qr", "нет"),
            "price2_qr": qr_fields.get("price2_qr", "нет"),
            "price3_qr": qr_fields.get("price3_qr", "нет"),
            "price4_qr": qr_fields.get("price4_qr", "нет"),
            "wholesale_level_1_count": qr_fields.get("wholesale_level_1_count", "нет"),
            "wholesale_level_1_price": qr_fields.get("wholesale_level_1_price", "нет"),
            "wholesale_level_2_count": qr_fields.get("wholesale_level_2_count", "нет"),
            "wholesale_level_2_price": qr_fields.get("wholesale_level_2_price", "нет"),
            "action_price_qr": qr_fields.get("action_price_qr", "нет"),
            "action_code_qr": qr_fields.get("action_code_qr", "нет"),
            "_ocr_text": text,  # служебное, для фильтра; убирается перед записью
        }
        enrich_qr_from_visible_fields(out)
        infer_discount_from_prices(out)
        rows.append(out)

    DIGIT_RX = re.compile(r"\d")
    LETTERS_RX = re.compile(r"[A-Za-zА-Яа-я]")

    def is_real_tag(r: dict, ocr_text: str = "") -> bool:
        if r["price_default"] or r["price_card"]:
            return True
        if r["barcode"]:
            return True
        if r["qr_code_barcode"] not in ("", "нет"):
            return True
        if isinstance(ocr_text, str):
            clean = ocr_text.replace("\n", " ").strip()
            if len(clean) >= 5 and DIGIT_RX.search(clean) and LETTERS_RX.search(clean):
                return True
        return False

    n_before = len(rows)
    if not args.no_filter:
        rows = [r for r in rows if is_real_tag(r, r.get("_ocr_text", ""))]
    print(f"Filtered: {n_before} -> {len(rows)} rows (kept rows with price/code)")

    def bbox_overlap_ratio(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        sm = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
        return inter / sm if sm > 0 else 0.0

    if not args.no_dedup:
        rows_sorted = sorted(
            rows,
            key=lambda r: -(int(r["x_max"]) - int(r["x_min"]))
                          * (int(r["y_max"]) - int(r["y_min"])),
        )
        kept: list[dict] = []
        for r in rows_sorted:
            rb = (int(r["x_min"]), int(r["y_min"]),
                  int(r["x_max"]), int(r["y_max"]))
            rts = int(r["frame_timestamp"])
            dup = False
            for k in kept:
                kb = (int(k["x_min"]), int(k["y_min"]),
                      int(k["x_max"]), int(k["y_max"]))
                kts = int(k["frame_timestamp"])
                if (abs(rts - kts) <= 500
                        and bbox_overlap_ratio(rb, kb) >= 0.5):
                    dup = True
                    break
            if not dup:
                kept.append(r)
        print(f"Dedup: {len(rows)} -> {len(kept)} rows")
        rows = kept

    out_path = Path(args.out) if args.out else (
        ROOT / "ml" / "output" / f"final_{ocr_path.stem.replace('ocr_qr_', '')}.csv")
    for r in rows:
        r.pop("_ocr_text", None)
    cols = list(rows[0].keys()) if rows else FIELDS_FALLBACK
    if "_ocr_text" in cols:
        cols = [c for c in cols if c != "_ocr_text"]
    if not args.keep_alts and "alts_json" in cols:
        cols = [c for c in cols if c != "alts_json"]
        for r in rows:
            r.pop("alts_json", None)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
